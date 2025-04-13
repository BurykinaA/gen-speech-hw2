from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from collections import defaultdict
import pandas as pd
import time
from pathlib import Path


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """
        arg_maxes = torch.argmax(logits, dim=1).tolist()
    
        decoded = []
        previous = self.blank_token_id
        for token_id in arg_maxes:
            if token_id != self.blank_token_id and token_id != previous:
                decoded.append(self.vocab[token_id])
            previous = token_id

        decoded_text = ''.join(decoded)
        decoded_text = ' '.join(decoded_text.split(self.word_delimiter))
        return decoded_text


    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        beams = [(0.0, [], self.blank_token_id)]

        T, V = logits.shape
        for t in range(T):
            current_log_probs = log_probs[t]
            new_beams = defaultdict(lambda: (-float('inf'), [], self.blank_token_id))
            for beam_score, beam_seq, last_token in beams:
                for v in range(V):
                    token = v
                    log_p = current_log_probs[v].item()
                    new_score = beam_score + log_p
                    
                    if token == self.blank_token_id:
                        key = (tuple(beam_seq), last_token)
                        if new_score > new_beams[key][0]:
                            new_beams[key] = (new_score, beam_seq.copy(), last_token)
                    elif token == last_token:
                        key = (tuple(beam_seq), token)
                        if new_score > new_beams[key][0]:
                            new_beams[key] = (new_score, beam_seq.copy(), token)
                    else:
                        new_seq = beam_seq.copy()
                        new_seq.append(token)
                        key = (tuple(new_seq), token)
                        if new_score > new_beams[key][0]:
                            new_beams[key] = (new_score, new_seq, token)
            
            beams = sorted(new_beams.values(), key=lambda x: x[0], reverse=True)[:self.beam_width]

        
        if return_beams:
            return [(seq, score) for score, seq, _ in beams]
        else:
            best_seq = beams[0][1] if beams else []
            decoded_text = ''.join([self.vocab[token] for token in best_seq])
            decoded_text = ' '.join(decoded_text.split(self.word_delimiter))
            return decoded_text

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
        
        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        beams = [(0.0, [], self.blank_token_id, 0)]

        T, V = logits.shape
        for t in range(T):
            current_log_probs = log_probs[t]
            new_beams = defaultdict(lambda: (-float('inf'), [], self.blank_token_id))
            for beam_score, beam_seq, last_token, word_count in beams:
                for v in range(V):
                    token = v
                    log_p = current_log_probs[v].item()
                
                    current_text = ''.join([self.vocab[t] for t in beam_seq])
                    if token != self.blank_token_id:
                        new_text = current_text + self.vocab[token]
                    else:
                        new_text = current_text
                    
                    lm_score = self.lm_model.score(new_text, bos=False, eos=False)
                    new_score = beam_score + log_p + self.alpha * lm_score
                    
                    if token == self.blank_token_id:
                        key = (tuple(beam_seq), last_token)
                        if new_score > new_beams[key][0]:
                            new_beams[key] = (new_score, beam_seq.copy(), last_token, word_count)
                    elif token == last_token:
                        key = (tuple(beam_seq), token)
                        if new_score > new_beams[key][0]:
                            new_beams[key] = (new_score, beam_seq.copy(), token, word_count)
                    else:
                        new_seq = beam_seq.copy()
                        new_seq.append(token)
                        new_word_count = word_count + (1 if self.vocab[token] == self.word_delimiter else 0)
                        new_score += self.beta
                        key = (tuple(new_seq), token)
                        if new_score > new_beams[key][0]:
                            new_beams[key] = (new_score, new_seq, token, new_word_count)
            
            beams = sorted(new_beams.values(), key=lambda x: x[0], reverse=True)[:self.beam_width]
    
        best_seq = beams[0][1] if beams else []
        decoded_text = ''.join([self.vocab[token] for token in best_seq])
        decoded_text = ' '.join(decoded_text.split(self.word_delimiter))
        return decoded_text

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
        
        Returns:
            str: Best rescored transcript
        """
        # print(beams)
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        rescored_beams = []
        for seq, log_prob in beams:
            text = ''.join([self.vocab[token] for token in seq])
            lm_score = self.lm_model.score(text, bos=False, eos=False)
            word_count = text.count(self.word_delimiter)
            combined_score = log_prob + self.alpha * lm_score + self.beta * word_count
            rescored_beams.append((combined_score, seq))
        
        best_seq = max(rescored_beams, key=lambda x: x[0])[1] if rescored_beams else []
        decoded_text = ''.join([self.vocab[token] for token in best_seq])
        decoded_text = ' '.join(decoded_text.split(self.word_delimiter))
        return decoded_text

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        #logits shape - (seq_len; vocabulary_len)

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription, results_file="decoding_results.csv"):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    results = []

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]: #"greedy", "beam", 
        print("-" * 60)
        print(f"{d_strategy} decoding")

        start_time = time.time()
        transcript = decoder.decode(audio_input, method=d_strategy)
        elapsed_time = time.time() - start_time

        clean_transcript = transcript.strip()
        lev_distance = Levenshtein.distance(true_transcription, clean_transcript)

        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {lev_distance}")
        
        results.append({
            'audio_path': audio_path,
            'method': d_strategy,
            'lev_distance': lev_distance,
            'time_sec': elapsed_time
        })
    
    df = pd.DataFrame(results)

    if Path(results_file).exists():
        existing_df = pd.read_csv(results_file)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(results_file, index=False)

if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]

    # beam_widths = [3, 5, 7]

    # for i in beam_widths:
    #     decoder = Wav2Vec2Decoder(beam_width = i)
    #     _ = [test(decoder, audio_path, target, results_file=f'beam_widths_{i}.csv') for audio_path, target in test_samples]

    # alphas = [0.1, 0.25, 0.5]

    # for i in alphas:
    #     decoder = Wav2Vec2Decoder(alpha = i)
    #     _ = [test(decoder, audio_path, target, results_file=f'alphas_{i}.csv') for audio_path, target in test_samples]

    # betas = [0.1, 0.5, 2.0, 5.0]

    # for i in betas:
    #     decoder = Wav2Vec2Decoder(alpha = 0.1, beta = i)
    #     _ = [test(decoder, audio_path, target, results_file=f'betas_{i}.csv') for audio_path, target in test_samples]
