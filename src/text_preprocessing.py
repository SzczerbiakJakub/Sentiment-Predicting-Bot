
class TextPreprocessing:

    stop_words_list = []
    stop_word_file_path = r".\datasets\eng_stop_words.txt"

    @staticmethod
    def get_stop_words_list():
        with open(TextPreprocessing.stop_word_file_path, 'r') as file:
            for line in file:
                y = line.split()
                for i in y:
                    TextPreprocessing.stop_words_list.append(i)

    @staticmethod
    def remove_quote_character(input):
        if "'" not in input and "." not in input:
            return input
        else:
            return "".join(x for x in input if x != "'" or x != ".")
        #input.replace(x, "")
        
    @staticmethod
    def remove_special_characters(input):
        input = input.replace("'", "")
        input = input.replace(".", "")
        input = input.replace(",", "")
        input = input.replace("!", "")
        input = input.replace("?", "")
        input = input.replace("/", "")
        input = input.replace('"', '')
        return input

    @staticmethod
    def remove_user_tags(input, input_words):
        if '@' in input:
            input_words.remove(input)

    @staticmethod
    def remove_stop_words(input, input_words):
        if input in TextPreprocessing.stop_words_list:
            input_words.remove(input)

    @staticmethod
    def preprocess_text(input):
        input = input.lower()
        input = TextPreprocessing.remove_special_characters(input)
        input_words = input.split()
        for x in input_words.copy():
            TextPreprocessing.remove_user_tags(x, input_words)
            TextPreprocessing.remove_stop_words(x, input_words)
        return " ".join(x for x in input_words)