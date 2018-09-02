import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    Xlengths = test_set.get_all_Xlengths()

    for word_id in Xlengths:
      
      prob_dict = {}
      best_score = float('-Inf')

      for model_word in models:

        try:
          model = models[model_word]
          logL = model.score(Xlengths[word_id][0], Xlengths[word_id][1])
          prob_dict[model_word] =  logL
        except:
          logL = float('-Inf')
          prob_dict[model_word] =  logL

        if logL > best_score:
          best_score = logL
          guess_word = model_word

      probabilities.append(prob_dict)
      guesses.append(guess_word)

    return probabilities, guesses


