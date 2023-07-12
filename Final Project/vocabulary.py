from collections import defaultdict

LEXICON_PATH = 'lexicon.txt'
TOKENS_PATH = 'tokens.txt'


class Vocabulary:
    def __init__(self, transcriptions, type='unigram'):
        self.transcriptions = transcriptions
        self.translator = self.create_trans_dict()
        self.size = len(self.translator)
        # self.generate_lexicon_file()
        # self.generate_tokens_file()
        # if type == 'unigram':
        #     self.language_model = self.create_unigram_lang_model()

    def create_trans_dict(self):
        """
        Given all transcriptions in the data, create a translation dictionary
        from each letter to a number.
        """
        vocabulary = set()
        for transcription in self.transcriptions:
            vocabulary.update(transcription)
        vocabulary.add('')

        vocabulary = sorted(list(vocabulary))
        return {letter: i for i, letter in enumerate(vocabulary)}

    def generate_lexicon(self):
        """
        Given all transcriptions in the data, create a lexicon.
        """
        unique_lines = set()  # Keep track of unique lines

        with open(LEXICON_PATH, 'w') as lexicon_file:
            for transcription in self.transcriptions:
                tokens = transcription.split()
                for token in tokens:
                    letters = ' '.join(list(token))
                    line = f"{token} {letters} |"
                    unique_lines.add(line)

            for line in unique_lines:
                lexicon_file.write(line)
                lexicon_file.write('\n')

    def generate_tokens_file(self):
        with open(TOKENS_PATH, 'w') as file:
            for char in self.translator.keys():
                file.write(char+'\n')

    def create_unigram_lang_model(self):
        """
        Given all transcriptions in the data, Create a unigram language model.
        """
        vocabulary = defaultdict(int)

        # Iterate through the text files and collect unique letters
        for transcription in self.transcriptions:
            for char in transcription:
                vocabulary[char] += 1

        total_repetitions = sum(vocabulary.values())
        return {letter: count / total_repetitions
                for letter, count in self.vocabulary.items()}
