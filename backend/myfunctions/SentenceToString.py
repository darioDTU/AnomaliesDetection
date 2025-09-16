def SentenceToString(sentence):
    """
    This function convert a Sentence to a list of string, it's used to identify differents error with the same error title

    In :
        -> Sentence
    Out :
        -> List of string
    """
    sentence = str(sentence)
    res = sentence.split()
    return res