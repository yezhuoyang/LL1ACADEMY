from LL1_Academy.tools import GrammarChecker,SingleGrammarGenerator, SvmLearn

import os

#Description: The MassGrammarGenerator class randomly generates grammar and filter
#out the interesting ones to store in the database. 

script_dir    = os.path.dirname(__file__) #<-- absolute dir the script is in
good_examples = os.path.join(script_dir, 'trainingData/'+str(n)+'var-interesting')
bad_examples  = os.path.join(script_dir, 'trainingData/'+str(n)+'var-not-interesting')

model = SvmLearn.SvmLearn(n,good_examples,bad_examples)

class
