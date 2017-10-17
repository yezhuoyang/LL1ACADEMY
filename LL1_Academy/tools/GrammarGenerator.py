# -*- coding: utf-8 -*-

from enum import Enum
import string
import numpy as np
import random
import GrammarClassifier
import logging

logger = logging.getLogger('GrammarGenerator')

epsilon = u'ε'

# Production templates
class RandomSeq:
  # choices: an array of the possible choices
  # 
  def __init__(self,choices,minLen,maxLen):
    self.choices = choices
    self.minLen = minLen
    self.maxLen = maxLen

  def generate(self):
    l = random.randint(self.minLen, self.maxLen)

    s = ""
    for i in range(0,l):
      s += random.choice(self.choices)
    return s

def randomSeq(choices,minLen,maxLen):
    l = random.randint(minLen, maxLen)

    s = ""
    for i in range(0,l):
      s += random.choice(choices)
    return s

def oneOf(choices):
  return randomSeq(choices,1,1)
def twoOf(choices):
  return randomSeq(choices,2,2)
def threeOf(choices):
  return randomSeq(choices,3,3)
def zeroOrMore(choices):
  return randomSeq(choices,0,3)
def oneOrMore(choices):
  return randomSeq(choices,1,3)

# Nonterminal template format:
# For each nonterminal, there can be multiple lines.
# Each line is a choice for the productions of that nonterminal.
# Each line defines multiple productions for that nonterminal.
# Productions can be reordered.
#
# A = P1 | P2 | ...
# A = P1 | P2 | ...
# ...
# B = P1 | P2 | ...


# Production template format:
# * : 0 or more terminals
# + : 1 or more terminals
# 1 : 1 terminal
# 2 : 2 terminals
# 3 : 3 terminals
# N : any nonterminal
# O : any other nonterminal
# <upper case letter> : specific nonterminal
# <lower case letter> : specific terminal

def strips(x):
  if type(x) is str:
    return x.strip()
  if type(x) is list:
    return map(strips,x)
  raise Error('strips: unknown arg type ' + type(x))

class GrammarGenerator:
  def __init__(self,terminals,src):
    self.terminals = terminals
    self.template = {}
    self.parse(src)
    self.init_permutation()

  def init_permutation(self):  
    # permute the terminal and nonterminals
    self.permute = {}
    nts = self.template.keys()
    nts_perm = np.random.permutation(nts)
    for i in range(0,len(nts)):
      self.permute[nts[i]] = nts_perm[i]

    terminals_perm = np.random.permutation(self.terminals)
    for i in range(0,len(self.terminals)):
      self.permute[self.terminals[i]] = terminals_perm[i]

    logger.info('init_permutation: ' + str(self.permute))

  def parse(self, src):
    lns = strips(src.split('\n'))
    for ln in lns:
      [nt,ps] = strips(ln.split('::='))
      ps = strips(ps.split('|'))
      if nt in self.template:
        self.template[nt].append(ps)
      else:
        self.template[nt] = [ps]

  def generate(self):
    done = False
    while not done:
      self.init_permutation()
      g = dict([(self.permute[nt],self.generateNonterminal(nt)) for nt in self.template])
      logger.info('generated grammar %s' % g)
      analyzer = GrammarClassifier.GrammarAnalyzer(g)
      done = True
      for nt in g:
        if nt == 'A':
          continue
        if not analyzer.reachable(nt,'A'):
          done = False
    return (g,analyzer)

  def generateNonterminal(self,nt):
    ps = np.random.permutation(random.choice(self.template[nt]))
    return [self.generateProduction(nt,p) for p in ps]
  
  def generateProduction(self,nt,p):
    r = ''.join([self.generateProductionPart(nt,p[i]) for i in range(0,len(p))])
    if r == '':
      return epsilon
    return r

  def generateProductionPart(self,nt,c):
    nts = self.template.keys()
    other_nts = filter(lambda x: x != nt, nts)
    handlers = {
      '+' : lambda : oneOrMore(self.terminals),
      '*' : lambda : zeroOrMore(self.terminals),
      '1' : lambda : oneOf(self.terminals),
      '2' : lambda : twoOf(self.terminals),
      '3' : lambda : threeOf(self.terminals),
      'N' : lambda : oneOf(nts),
      'O' : lambda : oneOf(other_nts)
    }
    if c in handlers:
      return handlers[c]()
    elif c in string.whitespace:
      return ""
    else:
      # specific terminal or nonterminal symbol
      return self.permute[c]


# Template internal representation:
# dictionary from nonterminals to arrays (sets of
# productions) of arrays (productions).

# Grammar representation:
# dictionary from nonterminals to arrays of strings

def renderGrammar(g):
  ls = []
  nts = g.keys()
  nts.sort()
  for nt in nts:
    ps = map(lambda p: p if len(p) > 0 else 'EPSILON', g[nt])
    r = string.join(ps, ' | ')
    ls.append('%s := %s' % (nt,r))
  return string.join(ls, '\n')

wxyz = [c for c in 'wxyz']

grammar1 = """
A ::= +B | 2OO | 
B ::= + | * 
C ::= +A | 2N1 | B
""".strip()

# Not LL1 because y is in first of both of A's productions
midterm03q2 = """
A ::= BCx | y
B ::= yA | 
C ::= Ay | x
""".strip()

midterm04q1 = """
A ::= ABx | y
B ::= x | 
""".strip()

midterm04q2 = """
A ::= BC | 
B ::= yAx | 
C ::= By | z
""".strip()

midterm04q3 = """
A ::= Bx
B ::= yC
C ::= x | Bx
""".strip()

midterm05q1 = """
A ::= Bx | By
B ::= Ay | x
""".strip()

midterm05q2 = """
A ::= xB | Cz
B ::= AxC
C ::= yB |
""".strip()

class QuestionGenerator:
  def __init__(self, grammar, analysis):
    self.grammar = grammar
    self.analysis = analysis
    self.nonterminals = self.grammar.keys()
    self.nonterminals.sort()
    self.terminals = self.analysis.terminals
    self.terminals.sort()

  def questions(self):
    xml = []
    xml.append('<group>')
    xml.append(self.example())
    xml.append(self.nullableQuestion())
    xml.append(self.firstSetQuestions())
    xml.append(self.followSetQuestions())
    xml.append('</group>')
    
    return '\n'.join(xml)

  def example(self):
    
    xml = []
    xml.append('<example>')
    xml.append(""" Consider the grammar %(grammar)s where %(nonterminals)s is the set
    of nonterminal symbols, $A$ is the start symbol, %(terminals)s is
    the set of terminal symbols, and $\epsilon$ denotes the empty
    string.""" % {
      'grammar' : self.renderGrammarLatex(),
      'nonterminals' : '\{%s\}' % ','.join(self.nonterminals),
      'terminals' : '\{%s\}' % ','.join(self.terminals)
    })
    xml.append('</example>')
    
    return '\n'.join(xml)
  
  def renderGrammarLatex(self):
    def renderProduction(p):
      if p == epsilon or p == '':
        return '$\epsilon$'
      return '$%s$' % ('\\ '.join([c for c in p]))
    
    def renderProductions(ps):
      # \v is vertical tab, so we have to escape
      return ' $\\verybigmid$ '.join([renderProduction(p) for p in ps])

    return '\n'.join([
      '\\begin{tabbing}',
      ' \\\\\n '.join([
        '$%s$ \= $\cce$ \= %s' % (nt,renderProductions(self.grammar[nt]))
        for nt in self.nonterminals
      ]),
      '\\end{tabbing}'
      ])

  def question(self, what, answers):
    xml = []
    xml.append('<question>')
    xml.append('<what>%s</what>' % what)
    xml.extend(answers)
    xml.append('</question>')
    return '\n'.join(xml)
  
  def nullableQuestion(self):
    return self.question(
      'Which nonterminals are nullable?',
      self.nullableAnswers()
    )
  def nullableAnswers(self):
    correct = list(self.analysis.nullable)
    correct.sort()
    
    incorrects = []

    while len(incorrects) < 3:
      n = np.random.choice(len(self.nonterminals))
      a = list(np.random.choice(self.nonterminals, n, False))
      a.sort()
      if a == correct or (a in incorrects):
        continue
      incorrects.append(a)
    
    return self.renderAnswers(correct, incorrects)
    
  def firstSetQuestions(self):
    qs = []
    for nt in self.nonterminals:
      qs.append(self.firstSetQuestion(nt))
    return '\n'.join(qs)

  def firstSetQuestion(self,nt):
    return self.question(
      'What is First($%s$)?' % nt,
      self.firstSetAnswers(nt)
    )

  def followSetQuestions(self):
    qs = []
    for nt in self.nonterminals:
      qs.append(self.followSetQuestion(nt))
    return '\n'.join(qs)

  def followSetQuestion(self,nt):
    return self.question(
      'What is Follow($%s$)?' % nt,
      self.followSetAnswers(nt)
    )

  def followSetAnswers(self,nt):
    log = logging.getLogger('QuestionGenerator.followSetAnswers')
    answers = {}

    correct = list(self.analysis.followSets[nt])
    correct.sort()
    correct_k = GrammarClassifier.concatStr(correct)
    log.debug('correct: %s' % correct_k)
    answers[correct_k] = correct

    follow_terms = self.analysis.terminals + ['$']
    
    while len(answers) < 4:
      n = np.random.choice(len(follow_terms))
      a = np.random.choice(follow_terms, n, False)
      a.sort()
      k = GrammarClassifier.concatStr(a)
      log.debug('incorrect: %s' % k)
      if k not in answers:
        answers[k] = a

    del answers[correct_k]
    return self.renderAnswers(correct, answers.values())
    

  def trueAnswer(self, answer):
    return '<answer true="true">$\{%s\}$</answer>' % ','.join(answer)
    
  def falseAnswer(self, answer):
    return '<answer>$\{%s\}$</answer>' % ','.join(answer)

  def renderAnswers(self, correct, incorrects):
    # replace '$' with '\\$' in a set of symbols
    def fixDollar(cs):
      return ['\\$' if c == '$' else c for c in cs]
    
    xml = []
    xml.append(self.trueAnswer(fixDollar(correct)))
    for a in incorrects:
      xml.append(self.falseAnswer(fixDollar(a)))
    return np.random.permutation(xml)
      
  def firstSetAnswers(self,nt):
    answers = {}

    correct = list(self.analysis.firstSets[nt])
    correct.sort()
    correct_k = GrammarClassifier.concatStr(correct)
    answers[correct_k] = correct
    
    while len(answers) < 4:
      n = np.random.choice(len(self.analysis.terminals))
      a = np.random.choice(self.analysis.terminals, n, False)
      a.sort()
      k = GrammarClassifier.concatStr(a)
      if k not in answers:
        answers[k] = a

    del answers[correct_k]
    return self.renderAnswers(correct, answers.values())

def generateExam(year, quarter, month, date, questionGroups):
  vars = {
    'year' : year,
    'quarter' : quarter,
    'month' : month,
    'date' : date,
    'questionGroups' : '\n'.join(questionGroups)
  }

  return """
<test>
<preamble>
\\def\\sem#1{\\mbox{$[\\hspace{-0.15em}[$}#1\\mbox{$]\\hspace{-0.15em}]$}} \\def\\bigwedge{\\;\\wedge\\;} \\newcommand{\\irule}[2]{\\mkern-2mu\\displaystyle\\frac{#1}{\\vphantom{,}#2}\\mkern-2mu} \\def\\air{\\hspace*{1.0cm}} \\def\\imp{\\rightarrow} \\def\\bigmid{\\;\\mid\\;} \\def\\verybigmid{\\;\\bigmid\\;} \\def\\cce{\\;\\; ::= \\;\\;}
</preamble>
<frontpage>
\\begin{tabbing} {\\Huge CS 132 Compiler Construction, %(quarter)s %(year)s} \\\\ \\\\ {\\Huge Instructor: Jens Palsberg} \\\\ \\\\ {\\Huge Multiple Choice Exam, %(month)s %(date)s, %(year)s} \\\\ \\\\ {\\LARGE ID~~~~~~}\\={\\LARGE \\raisebox{-4mm}{\\framebox[10cm]{\\rule{0mm}{10mm}}}}\\\\ \\\\ {\\LARGE Name}\\>{\\LARGE \\raisebox{-4mm}{\\framebox[10cm]{\\rule{0mm}{10mm}}}}\\\\ \\end{tabbing} \\bigskip \\bigskip This exam consists of 22 questions. Each question has four options, exactly one of which is correct, while the other three options are incorrect. For each question, you can check multiple options. \\bigskip \\bigskip I will grade each question in the following way. If you check {\\em none\\/} of the options, you get 0 points. If you check all {\\em four\\/} options, you get 0 points. {\\bf Check one option.} If you check {\\em one\\/} option, and that option is correct, you get 2 points. If you check {\\em one\\/} option, and that option is wrong, you get --0.667 points (yes, negative!). {\\bf Check two options.} If you check {\\em two\\/} options, and one of those options is correct, you get 1 point. If you check {\\em two\\/} options, and both of them are wrong, you get --1 point (yes, negative!). {\\bf Check three options.} If you check {\\em three\\/} options, and one of those options is correct, you get 0.415 points. If you check {\\em three\\/} options, and all three of them are wrong, you get --1.245 points (yes, negative!). \\bigskip \\bigskip The maximum point total is $22 \\times 2 = 44$ points. I will calculate a percentage based on the points in the following way: \\[ \\irule{\\max(0, \\mbox{point total})} {44} \\times 100 \\] Notice that if your point total is negative, you will get 0 percent.
</frontpage>
%(questionGroups)s
</test>
""" % vars

def test(n):
  questionGroups = []
  for i in range(n):
    gen = GrammarGenerator(wxyz, midterm05q2)
    (g,analyzer) = gen.generate()
    qGen = QuestionGenerator(g,analyzer) 
    questionGroups.append(qGen.questions())
  print generateExam("2017", "Fall", "Nov", "12", questionGroups)

def newTest():
  questionGroups = []
  for spec in ['EASY_LL1', 'EASY_NON_LL1', 'MED_LL1',
               'MED1_NON_LL1', 'MED2_NON_LL1',
               'HARD_LL1', 'HARD_NON_LL1']:
    g = GrammarClassifier.findGrammar(spec)
    a = GrammarClassifier.GrammarAnalyzer(g)
    qGen = QuestionGenerator(g,a)
    questionGroups.append(qGen.questions())
  print generateExam("2017", "Fall", "Nov", "12", questionGroups)

def generateGrammars(n=3):
  questionGroups = []
  for spec in ['EASY_LL1', 'EASY_NON_LL1', 'MED_LL1',
               'MED1_NON_LL1', 'MED2_NON_LL1',
               'HARD_LL1', 'HARD_NON_LL1']:
    title = str(GrammarClassifier.grammarSpecs[spec])
    print '=' * len(title)
    print title
    print '=' * len(title)
    
    for i in range(n):
      g = GrammarClassifier.findGrammar(spec)
      if g:
        a = GrammarClassifier.GrammarAnalyzer(g)
        print renderGrammar(g)
        print

  
if __name__ == '__main__':
  logging.basicConfig()
  #logging.getLogger().setLevel(logging.DEBUG)
  #test(5)
  #newTest()
  generateGrammars(3)
