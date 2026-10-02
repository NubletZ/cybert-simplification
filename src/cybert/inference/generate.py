"""Decoding routines.

Four modes, each serving a distinct role:

``greedy_decode``
    Hard ids, no gradient. Produces the stop-gradient pseudo-inputs ``y_hat``
    and ``x_hat`` of Eq. 11, and the fake samples for the discriminator update.

``soft_decode_teacher_forced``
    Differentiable. The second cycle pass runs teacher-forced against the
    original sentence and emits a distribution per step; those distributions
    become soft embeddings for phi (Eq. 9). This is ``cycle_mode:
    teacher_forced``.

``soft_decode_free_running``
    Differentiable free-running decoding with optional straight-through
    Gumbel-softmax. This is ``cycle_mode: gumbel``, and it also produces the
    fake sentence for the *generator* half of the adversarial loss (Eqs. 13,
    15), which needs a gradient path into G/F.

``beam_search``
    Inference only; Appendix G.2 uses ``k = 4``.

Soft outputs are ``(B, T, V)`` distributions. Multiplying one by an embedding
table gives an embedding sequence that BERT accepts through ``inputs_embeds``,
which is how a generated sentence reaches phi or a discriminator without an
argmax severing the graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.generator import CyBERTGenerator, Encoded


def _finished_mask(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """True where a sequence has already emitted EOS at an earlier step."""
    return (tokens == eos_id).cumsum(dim=1).gt(0)


def build_attention_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (ids != pad_id).long()


@torch.no_grad()
def greedy_decode(
    generator: CyBERTGenerator,
    encoded: Encoded,
    style: torch.Tensor | None = None,
    max_length: int = 64,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Greedy decode to hard ids of shape ``(B, T)``, EOS-terminated and padded."""
    state, memory = generator.prepare_decoding(encoded, style)
    batch = encoded.content.size(0)
    device = encoded.content.device
    memory_mask = encoded.attention_mask

    current = torch.full((batch,), generator.bos_id, dtype=torch.long, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    outputs: list[torch.Tensor] = []

    for _ in range(max_length):
        embedded = generator.decoder.embed(current)
        logits, state = generator.decoder.step(embedded, state, memory, memory_mask)
        if temperature != 1.0:
            logits = logits / temperature
        current = logits.argmax(dim=-1)
        current = torch.where(done, torch.full_like(current, generator.pad_id), current)
        outputs.append(current)
        done = done | current.eq(generator.eos_id)
        if bool(done.all()):
            break

    return torch.stack(outputs, dim=1)


def soft_decode_teacher_forced(
    generator: CyBERTGenerator,
    encoded: Encoded,
    decoder_input_ids: torch.Tensor,
    style: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable ``(B, T, V)`` distributions from a teacher-forced pass."""
    logits = generator.decode_teacher_forced(encoded, decoder_input_ids, style)
    return F.softmax(logits / temperature, dim=-1)


def soft_decode_free_running(
    generator: CyBERTGenerator,
    encoded: Encoded,
    style: torch.Tensor | None = None,
    max_length: int = 64,
    tau: float = 1.0,
    gumbel: bool = True,
    hard: bool = True,
) -> torch.Tensor:
    """Differentiable free-running decode, returning ``(B, T, V)``.

    With ``gumbel=True, hard=True`` the forward pass carries a one-hot vector
    while the backward pass sees the relaxed distribution (straight-through),
    so the sequence the discriminator scores is a real token sequence and the
    gradient still reaches the generator.
    """
    state, memory = generator.prepare_decoding(encoded, style)
    batch = encoded.content.size(0)
    device = encoded.content.device
    memory_mask = encoded.attention_mask

    current = torch.full((batch,), generator.bos_id, dtype=torch.long, device=device)
    embedded = generator.decoder.embed(current)
    steps: list[torch.Tensor] = []

    for _ in range(max_length):
        logits, state = generator.decoder.step(embedded, state, memory, memory_mask)
        if gumbel:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            probs = F.softmax(logits / tau, dim=-1)
        steps.append(probs)
        # Feed the soft symbol straight back in; the graph stays connected.
        embedded = generator.decoder.embed_soft(probs)

    return torch.stack(steps, dim=1)


def soft_to_embeddings(probs: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
    """``(B, T, V) @ (V, H) -> (B, T, H)``, for feeding BERT via inputs_embeds."""
    return probs @ embedding_weight


def prepend_cls_embedding(
    embeds: torch.Tensor, cls_embedding: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend a ``[CLS]`` embedding so pooled encoders see their pooling token.

    Returns ``(embeds, attention_mask)``. Pass ``mask`` when the soft sequence
    was produced by teacher forcing against a padded target, so the padded
    positions do not contribute to the pooled representation; free-running
    sequences have no padding and default to an all-ones mask.
    """
    batch, length, _ = embeds.shape
    cls = cls_embedding.view(1, 1, -1).expand(batch, 1, -1)
    out = torch.cat([cls, embeds], dim=1)
    if mask is None:
        mask = torch.ones(batch, length, dtype=torch.long, device=embeds.device)
    ones = torch.ones(batch, 1, dtype=mask.dtype, device=mask.device)
    return out, torch.cat([ones, mask], dim=1)


def _block_repeat_ngrams(
    logits: torch.Tensor, tokens: torch.Tensor, ngram_size: int
) -> torch.Tensor:
    """Set logits of tokens that would complete a repeated n-gram to -inf."""
    if ngram_size <= 0 or tokens.size(1) < ngram_size:
        return logits
    for row in range(tokens.size(0)):
        sequence = tokens[row].tolist()
        prefix = tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else ()
        banned = {
            sequence[i + ngram_size - 1]
            for i in range(len(sequence) - ngram_size + 1)
            if tuple(sequence[i : i + ngram_size - 1]) == prefix
        }
        for token in banned:
            logits[row, token] = torch.finfo(logits.dtype).min
    return logits


@torch.no_grad()
def beam_search(
    generator: CyBERTGenerator,
    encoded: Encoded,
    style: torch.Tensor | None = None,
    beam_size: int = 4,
    max_length: int = 64,
    temperature: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Batched beam search; returns the best hypothesis per input, ``(B, T)``."""
    if beam_size <= 1:
        return greedy_decode(generator, encoded, style, max_length, temperature)

    state, memory = generator.prepare_decoding(encoded, style)
    batch = encoded.content.size(0)
    device = encoded.content.device
    k = beam_size

    # (layers, B, H) -> (layers, B*k, H); memory (B, S, H) -> (B*k, S, H).
    h, c = state
    h = h.repeat_interleave(k, dim=1)
    c = c.repeat_interleave(k, dim=1)
    state = (h.contiguous(), c.contiguous())
    if memory is not None:
        memory = memory.repeat_interleave(k, dim=0)
    memory_mask = (
        encoded.attention_mask.repeat_interleave(k, dim=0)
        if encoded.attention_mask is not None
        else None
    )

    # Only the first beam of each item starts alive, so the k expansions of
    # step 1 are distinct rather than k copies of the same continuation.
    scores = torch.full((batch, k), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)  # (B*k,)

    tokens = torch.full((batch * k, 0), 0, dtype=torch.long, device=device)
    current = torch.full((batch * k,), generator.bos_id, dtype=torch.long, device=device)
    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(batch)]
    alive = torch.ones(batch * k, dtype=torch.bool, device=device)

    for step in range(max_length):
        embedded = generator.decoder.embed(current)
        logits, state = generator.decoder.step(embedded, state, memory, memory_mask)
        if temperature != 1.0:
            logits = logits / temperature
        if no_repeat_ngram_size and tokens.size(1) >= no_repeat_ngram_size:
            logits = _block_repeat_ngrams(logits, tokens, no_repeat_ngram_size)

        log_probs = F.log_softmax(logits, dim=-1)  # (B*k, V)
        log_probs = log_probs + scores.unsqueeze(1)
        log_probs = log_probs.masked_fill(~alive.unsqueeze(1), float("-inf"))

        vocab = log_probs.size(-1)
        flat = log_probs.view(batch, k * vocab)
        top_scores, top_idx = flat.topk(k, dim=-1)  # (B, k)
        beam_idx = torch.div(top_idx, vocab, rounding_mode="floor")
        token_idx = top_idx % vocab

        # Reorder beams.
        global_beam = (torch.arange(batch, device=device).unsqueeze(1) * k + beam_idx).view(-1)
        tokens = torch.cat([tokens[global_beam], token_idx.view(-1, 1)], dim=1)
        state = (state[0][:, global_beam].contiguous(), state[1][:, global_beam].contiguous())
        scores = top_scores.view(-1)
        alive = alive[global_beam]
        current = token_idx.view(-1)

        # Retire hypotheses that emitted EOS.
        ended = current.eq(generator.eos_id) & alive
        for pos in ended.nonzero(as_tuple=False).view(-1).tolist():
            item = pos // k
            penalty = ((step + 1) ** length_penalty) if length_penalty != 0 else 1.0
            finished[item].append((float(scores[pos]) / penalty, tokens[pos].tolist()))
        alive = alive & ~ended
        scores = scores.masked_fill(~alive, float("-inf"))

        if not bool(alive.any()):
            break

    # Fall back to the best unfinished beam when nothing terminated in time.
    best: list[list[int]] = []
    for item in range(batch):
        if finished[item]:
            best.append(max(finished[item], key=lambda pair: pair[0])[1])
        else:
            offset = item * k
            row = int(scores[offset : offset + k].argmax())
            best.append(tokens[offset + row].tolist())

    width = max((len(seq) for seq in best), default=1)
    out = torch.full((batch, max(1, width)), generator.pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(best):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out


def strip_after_eos(ids: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    """Replace everything at and after the first EOS with padding."""
    after = _finished_mask(ids, eos_id)
    # cumsum marks the EOS position itself as finished; shift so EOS survives.
    shifted = torch.zeros_like(after)
    shifted[:, 1:] = after[:, :-1]
    return ids.masked_fill(shifted, pad_id)
