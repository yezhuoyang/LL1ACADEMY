#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
import string
import numpy as np
import random
import GrammarClassifier
import logging
import argparse
import sys
import cmd
from prettytable import PrettyTable

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
    return list(map(strips,x))
  raise Error('strips: unknown arg type ' + type(x))

def parseGrammar(lns):
  g = {}
  for ln in lns:
    [nt,ps] = strips(ln.split('::='))
    ps = strips(ps.split('|'))
    g[nt]=ps
  return g

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
  nts = list(g.keys())
  nts.sort()
  for nt in nts:
    ps = map(lambda p: p if len(p) > 0 else 'EPSILON', g[nt])
    r = ' | '.join(ps)
    ls.append('%s ::= %s' % (nt,r))
  return '\n'.join(ls)

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

def renderSet(s):
  return '$\{%s\}$' % ','.join(['\\$' if c == '$' else c for c in s])

def renderSymbolCode(s):
  if s in string.ascii_lowercase:
    return 'eat(%s)' % s
  else:
    return '%s()' % s
    
def renderProdCode(p):
  if p == '':
    return '/* do nothing */'
  
  return ' '.join([renderSymbolCode(s) + ';' for s in p])

def renderProd(p):
  if p == epsilon or p == '':
    return '$\epsilon$'
  return '$%s$' % ('\\ '.join([c for c in p]))

def renderProds(ps):
  # \v is vertical tab, so we have to escape
  return ' $\\verybigmid$ '.join([renderProd(p) for p in ps])

def renderGrammarLatex(g):
  def sort(xs):
    xs = list(xs)
    xs.sort()
    return xs
  
  return '\n'.join([
    '\\begin{tabbing}',
    ' \\\\\n '.join([
      '$%s$ \= $\cce$ \= %s' % (nt,renderProds(g[nt]))
      for nt in sort(g.keys())
    ]),
    '\\end{tabbing}'
    ])

class QuestionGenerator:
  def __init__(self, grammar, analysis):
    self.grammar = grammar
    self.analysis = analysis
    self.nonterminals = list(self.grammar.keys())
    self.nonterminals.sort()
    self.terminals = self.analysis.terminals
    self.terminals.sort()

  def firstFollowQuestions(self):
    xml = []
    xml.append('<group>')
    xml.append(self.firstFollowExample())
    xml.append(self.nullableQuestion())
    xml.append(self.firstSetQuestions())
    xml.append(self.followSetQuestions())
    xml.append(self.isLL1Question())
    xml.append('</group>')
    return '\n'.join(xml)

    
  def firstFollowExample(self):
    xml = []
    xml.append('<example>')
    xml.append(""" Consider the grammar %(grammar)s where %(nonterminals)s is the set
    of nonterminal symbols, $A$ is the start symbol, %(terminals)s is
    the set of terminal symbols, and $\epsilon$ denotes the empty
    string.""" % {
      'grammar' : renderGrammarLatex(self.grammar),
      'nonterminals' : '\{%s\}' % ','.join(self.nonterminals),
      'terminals' : '\{%s\}' % ','.join(self.terminals)
    })
    xml.append('</example>')
    
    return '\n'.join(xml)

  def parseTableExample(self):
    xml = []
    xml.append('<example>')
    xml.append("""
Consider the grammar %(grammar)s where %(nonterminals)s is the set of nonterminal symbols, $A$ is the start symbol, %(terminals)s is the set of terminal symbols, and $\epsilon$ denotes the empty string. The grammar is LL(1). The predictive parsing table is a two-dimensional table called $table$.
""" % {
      'grammar' : renderGrammarLatex(self.grammar),
      'nonterminals' : '\{%s\}' % ','.join(self.nonterminals),
      'terminals' : '\{%s\}' % ','.join(self.terminals)
    })
    xml.append('</example>')
    return '\n'.join(xml)

  def parserExample(self):
    xml = []
    xml.append('<example>')
    xml.append("""
    Consider the grammar %(grammar)s where \{A,B,C\} is the set of nonterminal symbols, $A$ is the start symbol, %(terminals)s is the set of terminal symbols, and $\epsilon$ denotes the empty string. The grammar is LL(1). Assume that a recursive-descent parser for the above grammer declares a variable {\\tt next} of type {\\tt token}, and that the program has three procedures {\\tt A()}, {\\tt B()}, {\\tt C()}, and the following main part: \\begin{verbatim} void main() { next = getnexttoken(); A(); eat(EOF); } \end{verbatim} The procedure {\\tt getnexttoken()} gets the next token from an input file. The procedure {\\tt eat()} is here written in pseudo-code: \\begin{verbatim} void eat(token t) { if (t == next) { next = getnexttoken(); } else { error(); } \end{verbatim}""" % {
      'grammar' : renderGrammarLatex(self.grammar),
      'terminals' : '\{%s\}' % ','.join(self.terminals)
    })
    xml.append('</example>')
    
    return '\n'.join(xml)

  def parserQuestions(self):
    if self.nonterminals != ['A','B','C']:
      raise Error('parseTable questions are required to have nonterminals {A,B,C}.')
    
    if not self.analysis.isLL1:
      raise Error('parseTable questions require an LL1 grammar.')

    xml = []
    xml.append('<group>')
    xml.append(self.parserExample())

    for nt in self.nonterminals:
      for t in self.terminals:
        xml.append(self.parserQuestion(nt,t))

    xml.append('</group>')
    return '\n'.join(xml)

  def parserQuestion(self,nt,t):
    return self.question("""
The procedure {\\tt %(nt)s()} contains the snippet:
\\begin{verbatim}
  if (next == %(t)s) { ???? }
\\end{verbatim}
What is ``{\\tt ????}''?
""" % { 'nt' : nt, 't' : t},
    self.parserAnswers(nt,t)
    )

  def parserAnswers(self,nt,t):
    (correct,incorrect) = self.parseTableAnswersHelper(nt,t)
    def render(a):
      if(a == 'error'):
        return '{\\tt error();}'
      else:
        return '{\\tt %s}' % renderProdCode(a)
    return self.renderAnswers(render(correct), map(render,incorrect))
    
    

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
    
    return self.renderAnswers(renderSet(correct), map(renderSet,incorrects))
    
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

  def firstSetAnswers(self,nt):
    answers = {}

    correct = list(self.analysis.firstSets[nt])
    correct.sort()
    correct_k = ''.join(correct)
    answers[correct_k] = correct
    
    while len(answers) < 4:
      n = np.random.choice(len(self.analysis.terminals))
      a = np.random.choice(self.analysis.terminals, n, False)
      a.sort()
      k = ''.join(a)
      if k not in answers:
        answers[k] = a

    del answers[correct_k]
    return self.renderAnswers(renderSet(correct), map(renderSet,answers.values()))

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
    correct_k = ''.join(correct)
    log.debug('correct: %s' % correct_k)
    answers[correct_k] = correct

    follow_terms = self.analysis.terminals + ['$']
    
    while len(answers) < 4:
      n = np.random.choice(len(follow_terms))
      a = np.random.choice(follow_terms, n, False)
      a.sort()
      k = ''.join(a)
      log.debug('incorrect: %s' % k)
      if k not in answers:
        answers[k] = a

    del answers[correct_k]
    return self.renderAnswers(renderSet(correct), map(renderSet,answers.values()))

  def isLL1Question(self):
    answers = []
    if self.analysis.isLL1:
      answers.append(self.trueAnswer('Yes'))
      answers.append(self.falseAnswer('No'))
    else:
      answers.append(self.falseAnswer('Yes'))
      answers.append(self.trueAnswer('No'))

    answers.append(self.falseAnswer(
      'The question cannot be answered with the information provided.'
    ))
      
    answers.append(self.falseAnswer(
      'The LL(1)-checker would go into an infinite loop.'
    ))

    return self.question('Is the grammar LL(1)?', answers)
  
  def trueAnswer(self, answer):
    return '<answer true="true">%s</answer>' % answer
    
  def falseAnswer(self, answer):
    return '<answer>%s</answer>' % answer

  def renderAnswers(self, correct, incorrects):
    xml = []
    xml.append(self.trueAnswer(correct))
    for a in incorrects:
      xml.append(self.falseAnswer(a))
    return np.random.permutation(xml)
      
  def parseTableQuestions(self):
    if self.nonterminals != ['A','B','C']:
      raise Error('parseTable questions are required to have nonterminals {A,B,C}.')
    
    if not self.analysis.isLL1:
      raise Error('parseTable questions require an LL1 grammar.')

    xml = []
    xml.append('<group>')
    xml.append(self.parseTableExample())

    for nt in self.nonterminals:
      for t in self.terminals:
        xml.append(self.parseTableQuestion(nt,t))

    xml.append('</group>')
    return '\n'.join(xml)
  
  
  def parseTableQuestion(self, nt, t):
    return self.question(
      'What does table(%s,%s) contain?' % (nt, t),
      self.parseTableAnswers(nt,t)
    )

  def parseTableAnswersHelper(self,nt,t):
    correct = self.analysis.parseTable[nt].get(t,'error')

    if correct != 'error':
      # correct is a singleton set; extract the item
      correct = list(correct)[0]
      
    prods = self.grammar[nt] + ['error']
    prods.remove(correct)
      
    incorrect = list(np.random.choice(prods, min(3,len(prods)), False))
      
    # need to get more answers from somewhere. Choose productions
    # from other nonterminals I guess
    prods = []
    for ps in self.grammar.values():
      prods.extend(ps)

    while len(incorrect) < 3:
      p = np.random.choice(prods)
      if p not in incorrect and p != "" and p != correct:
        incorrect.append(p)
        
    return (correct,incorrect)

  def parseTableAnswers(self,nt,t):
    (correct,incorrect) = self.parseTableAnswersHelper(nt,t)
        
    def render(a):
      if a == 'error':
        return a
      else:
        return renderProd(a)
      
    return self.renderAnswers(render(correct),map(render,incorrect))
  
def generateExam(year, quarter, month, date, questionGroups):
  num_questions = 0
  for q in questionGroups:
    num_questions += q.count('<question>')
  
  vars = {
    'year' : year,
    'quarter' : quarter,
    'month' : month,
    'date' : date,
    'num_questions' : num_questions,
    'double_num_questions' : num_questions * 2,
    'questionGroups' : '\n'.join(questionGroups)
  }

  return """
<test>
<preamble>
\\def\\sem#1{\\mbox{$[\\hspace{-0.15em}[$}#1\\mbox{$]\\hspace{-0.15em}]$}} \\def\\bigwedge{\\;\\wedge\\;} \\newcommand{\\irule}[2]{\\mkern-2mu\\displaystyle\\frac{#1}{\\vphantom{,}#2}\\mkern-2mu} \\def\\air{\\hspace*{1.0cm}} \\def\\imp{\\rightarrow} \\def\\bigmid{\\;\\mid\\;} \\def\\verybigmid{\\;\\bigmid\\;} \\def\\cce{\\;\\; ::= \\;\\;}
</preamble>
<frontpage>
\\begin{tabbing} {\\Huge CS 132 Compiler Construction, %(quarter)s %(year)s} \\\\ \\\\ {\\Huge Instructor: Jens Palsberg} \\\\ \\\\ {\\Huge Multiple Choice Exam, %(month)s %(date)s, %(year)s} \\\\ \\\\ {\\LARGE ID~~~~~~}\\={\\LARGE \\raisebox{-4mm}{\\framebox[10cm]{\\rule{0mm}{10mm}}}}\\\\ \\\\ {\\LARGE Name}\\>{\\LARGE \\raisebox{-4mm}{\\framebox[10cm]{\\rule{0mm}{10mm}}}}\\\\ \\end{tabbing} \\bigskip \\bigskip This exam consists of %(num_questions)s questions. Each question has four options, exactly one of which is correct, while the other three options are incorrect. For each question, you can check multiple options.\\\\ \\\\ I will grade each question in the following way. 
\\begin{itemize}
\\item If you check {\\em none\\/} of the options, you get 0 points. 
\\item If you check all {\\em four\\/} options, you get 0 points. 
\\item {\\bf Check one option.} If you check {\\em one\\/} option, and that option is correct, you get 2 points. If you check {\\em one\\/} option, and that option is wrong, you get --0.667 points (yes, negative!). 
\\item {\\bf Check two options.} If you check {\\em two\\/} options, and one of those options is correct, you get 1 point. If you check {\\em two\\/} options, and both of them are wrong, you get --1 point (yes, negative!). 
\\item {\\bf Check three options.} If you check {\\em three\\/} options, and one of those options is correct, you get 0.415 points. If you check {\\em three\\/} options, and all three of them are wrong, you get --1.245 points (yes, negative!).
\\end{itemize}
The maximum point total is $%(num_questions)s \\times 2 = %(double_num_questions)s$ points. I will calculate a percentage based on the points in the following way: \\[ \\irule{\\max(0, \\mbox{point total})} {%(double_num_questions)s} \\times 100 \\] Notice that if your point total is negative, you will get 0 percent.
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
    questionGroups.append(qGen.firstFollowQuestions())
  print(generateExam("2017", "Fall", "Nov", "12", questionGroups))

specNames = ['EASY_LL1', 'EASY_NON_LL1', 'MED_LL1',
             'MED1_NON_LL1', 'MED2_NON_LL1',
             'HARD_LL1', 'HARD_NON_LL1']
  
def newTest():
  questionGroups = []
  for spec in specNames:
    g = GrammarClassifier.findGrammar(spec)
    a = GrammarClassifier.GrammarAnalyzer(g)
    qGen = QuestionGenerator(g,a)
    questionGroups.append(qGen.firstFollowQuestions())
  print(generateExam("2017", "Fall", "Nov", "12", questionGroups))

def generateGrammars(n=3):
  questionGroups = []
  for spec in ['EASY_LL1', 'EASY_NON_LL1', 'MED_LL1',
               'MED1_NON_LL1', 'MED2_NON_LL1',
               'HARD_LL1', 'HARD_NON_LL1']:
    title = str(GrammarClassifier.grammarSpecs[spec])
    print('=' * len(title))
    print(title)
    print('=' * len(title))
    
    for i in range(n):
      g = GrammarClassifier.findGrammar(spec)
      if g:
        a = GrammarClassifier.GrammarAnalyzer(g)
        print(renderGrammar(g))
        print()

def dictPrettyTable(d, keyTitle, valTitle):
  ks = list(d.keys())
  ks.sort()
  t = PrettyTable([keyTitle, valTitle])
  for k in ks:
    t.add_row([k,d[k]])
  return t
        
class InteractiveGenerator(cmd.Cmd):
  intro = 'Type help or ? to list commands.\n'
  prompt = '> '
  
  grammar = None
  analyzer = None
  
  def do_specs(self,arg):
    'Show the list of available specs'
    t = PrettyTable(['Name', 'Specification'])
    for s in specNames:
      t.add_row([s,str(GrammarClassifier.grammarSpecs[s])])
    print(t)

  def do_show(self,arg):
    'Show the last generated grammar.'
    print(self.grammar)
    
  def do_generate(self,arg):
    'Generate a grammar from a spec.\n    generate <spec name>'

    if arg == '':
      print('You must provide a valid spec name: ')
      self.do_specs('')
      return
    
    self.grammar = GrammarClassifier.findGrammar(arg)
    self.analyzer = GrammarClassifier.GrammarAnalyzer(self.grammar)
    print('Generated:\n' + renderGrammar(self.grammar))

  def do_nullable(self,arg):
    'Show the nullable nonterminals of the previously generated grammar.'
    if self.grammar == None:
      print('You must first generate a grammar.')
      return
    
    print('Nullable nonterminals:\n    ' + self.analyzer.nullable)

  def do_first(self,arg):
    'Show the first sets of the previously generated grammar.'
    if self.grammar == None:
      print('You must first generate a grammar.')
      return

    print(dictPrettyTable(self.analyzer.firstSets, 'Nonterminal', 'First Set'))
    
  def do_follow(self,arg):
    'Show the follow sets of the previously generated grammar.'
    if self.grammar == None:
      print('You must first generate a grammar.')
      return

    print(dictPrettyTable(self.analyzer.followSets, 'Nonterminal', 'Follow Set'))
    
    
  def do_parseTable(self,arg):
    'Show the parse table of the previously generated grammar.'
    if self.grammar == None:
      print('You must first generate a grammar.')
      return

    cols = ['Nonterminal'] + self.analyzer.terminals
    tbl = PrettyTable(cols)
    for nt in self.grammar:
      row = [nt]
      for t in self.analyzer.terminals:
        row.append(self.analyzer.parseTable[nt].get(t,'error'))
      tbl.add_row(row)
      
    print(tbl)

  def do_input(self,arg):
    """
    Manually input a grammar. Example:
    > input
    A ::= x | B
    B ::= y | C
    C ::= z | A

    (a blank line terminates the input)
    """

    lns=[]
    while True:
      ln = sys.stdin.readline().strip()
      if ln == "":
        break
      lns.append(ln)

    self.grammar = parseGrammar(lns)
    self.analyzer = GrammarClassifier.GrammarAnalyzer(self.grammar)
    print('Input successful.\n' + renderGrammar(self.grammar))
    
  def do_export(self,arg):
    """
    Export the exam to a file.

       export <filename> <year> <quarter> <month> <date> <question1> <question2> ... 
    """

    args = arg.split()
    examFile = args.pop(0)
    year = args.pop(0)
    quarter = args.pop(0)
    month = args.pop(0)
    date = args.pop(0)

    xml=[]
    for f in args:
      with open(f) as h:
        xml.append(h.read())

    with open(examFile,'w') as h:
      h.write(generateExam(year,quarter,month,date,xml))
    
  def do_save(self,arg):
    """
    Save the questions for a previously generated grammar in XML.
    The first parameter is the destination filename.
    The second parameter specifies which question types to generate.
    firstFollow generates questions about nullable nonterminals, and first 
    and follow sets.
    parseTable generates questions about the parse table.
    parser generates questions about the implementation of a parser in Java.

       save <filename> [firstFollow|table|parser]*
    """
    if self.grammar == None:
      print('You must first generate a grammar.')
      return

    args = arg.split()
    
    if len(args) < 2:
      print('You must provide both parameters.')
      return

    file=args[0]
    type=args[1]
    
    qGen = QuestionGenerator(self.grammar,self.analyzer)
    f = open(file,'w')
    if type == 'firstFollow':
      f.write(qGen.firstFollowQuestions())
    elif type == 'table':
      f.write(qGen.parseTableQuestions())
    elif type == 'parser':
      f.write(qGen.parserQuestions())
    else:
      print('Invalid question type.')
      
    f.write('\n')
    f.close()
        
if __name__ == '__main__':
  logging.basicConfig()

  argparser = argparse.ArgumentParser()
  argparser.add_argument('-n', type=int, metavar='N', default=1,
                         help='number of grammars to generate')
  argparser.add_argument('-d', action='store_true',
                         help='enable debug logging')
  argparser.add_argument('-s', choices=specNames, default=specNames[0],
                         help='specification for the grammar')
  argparser.add_argument('-o', type=str, metavar='FILENAME',
                         help='generate exam questions XML')

  args = argparser.parse_args(sys.argv[1:])

  if args.d:
    logging.getLogger().setLevel(logging.DEBUG)

  try:
    InteractiveGenerator().cmdloop()
  except KeyboardInterrupt as e:
    pass
    
  #test(5)
  #newTest()
  #generateGrammars(3)


