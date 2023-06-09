""" functions to generate random symbol """
import random
import string

def symbolize() -> str:
    """
    return string  (b24gh), -> decide which slots to change (categorical, time, number_of_people)
    """
    symbolized_text_list = []
    num_parts = random.choice([1])
    for part in range(num_parts):
        part_text_list = []
        num_integers = random.choice([2, 3, 4, 5])
        for integer in range(num_integers):
            part_text_list.append(str(random.randint(0, 9)))

        num_characters = random.choice([2, 3, 4, 5])
        for character in range(num_characters):
            part_text_list.append(random.choice(string.ascii_letters))
        random.shuffle(part_text_list)
        symbolized_text_list.append("".join(part_text_list))
    return " ".join(symbolized_text_list)