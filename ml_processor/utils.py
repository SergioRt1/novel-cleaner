import re



def split_into_sentences(text):
    # Regex to capture sentences along with their separators
    pattern = re.compile(r'(.*?)(\. |\n|\! |\” |\? |\" |\* |\] |』 )')
    matches = pattern.findall(text)
    sentences = []

    for sentence, separator in matches:
        full_sentence = sentence + separator
        sentences.append(full_sentence)

    # Handle any remaining text after the last separator
    remainder = pattern.sub('', text)
    if remainder:
        sentences.append(remainder)
    return sentences


if __name__ == "__main__":
    text = """“Theodore Miller.”
“However, you can’t graduate since you failed the practical requirement.” Vince’s decisive voice landed heavily on Theo’s shoulders. The academy had two graduation requirements. The first was to obtain a written examination score of above seventy points. The second was to become a 3rd?Circle master. The first wasn’t difficult, but the second requirement was a large problem.
It was just too difficult for Theo to become a 3rd?Circle master, especially as he was born with abysmal magic power and sensitivity. Even though he forwent sleep in order to squeeze in more practice, he couldn’t even reach the level of his classmates’ feet. Despite hours of practice, magic would always wrest out of his control.
As a result, Theo hadn’t been able to graduate for three years.
“Hmm…?Theodore, what circle are you now?” Professor Vince asked, as a tinge of frustration entered his voice."""
    split_into_sentences(text)