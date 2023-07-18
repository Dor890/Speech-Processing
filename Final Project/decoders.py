import torch
import torch.nn as nn


class BeamSearchDecoder:
    def __init__(self, model, beam_width):
        self.model = model
        self.beam_width = beam_width
        self.name = 'Beam'

    def decode(self, input_data, max_length):
        with torch.no_grad():
            input_data = torch.Tensor(input_data)
            output = self.model(input_data)

            output = nn.functional.log_softmax(output, dim=2)

            seq_probs = []
            seqs = [[]]

            for t in range(output.size(1)):
                if t >= max_length:
                    break

                curr_probs = output[:, t, :]
                curr_probs = curr_probs.repeat(len(seqs), 1, 1)

                seq_probs_t = []

                for i, seq in enumerate(seqs):
                    seq_prob = seq_probs[i]

                    if len(seq) > 0:
                        last_token = seq[-1]
                        curr_probs[i, :, last_token] = float('-inf')

                    top_probs, top_tokens = torch.topk(curr_probs[i],
                                                       self.beam_width)
                    top_probs = top_probs + seq_prob

                    seq_probs_t.extend(top_probs.flatten().tolist())

                topk_probs, topk_indices = torch.tensor(seq_probs_t).topk(
                    self.beam_width)

                seq_probs = []
                new_seqs = []

                for k in topk_indices:
                    beam_index = k // self.beam_width
                    token_index = k % self.beam_width

                    seq_probs.append(topk_probs[beam_index].item())
                    seq = seqs[beam_index] + [token_index]
                    new_seqs.append(seq)

                seqs = new_seqs

            best_seq_index = torch.argmax(torch.tensor(seq_probs))
            best_seq = seqs[best_seq_index]

            return best_seq


class GreedyDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.name = 'Greedy'

    def forward(self, emission: torch.Tensor):
        """
        Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors with shape [num_seq, num_label].
        Returns:
          List[str]: The resulting transcript.
        """
        # emission[:, :, :2] = -20
        indices = torch.argmax(emission, dim=2)  # [batch, num_seq]
        res = []
        for batch in indices:
            res.append("".join([self.labels[i] for i in batch if i != self.blank]))
        return res
