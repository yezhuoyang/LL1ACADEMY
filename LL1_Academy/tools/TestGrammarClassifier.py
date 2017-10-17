import unittest
from GrammarClassifier import *
from GrammarGenerator import renderGrammar
from prettytable import PrettyTable

def renderParseTable(parseTable):
  nts = parseTable.keys()
  nts.sort()

  # not all nts have entries for all ts, so we have to
  # union them all
  ts = set()
  for nt in nts:
    for ps in parseTable[nt].values():
      ts.update([t for p in ps for t in p if t in string.lowercase])
  ts = list(ts)
  ts.sort()

  tbl = PrettyTable([""] + ts)
  for nt in nts:
    r = [nt]
    for t in ts:
      if t in parseTable[nt]:
        r.append(list(parseTable[nt][t]))
      else:
        r.append("Error")
    tbl.add_row(r)
  return str(tbl)

def grammarSpecs(g):
  if g == None:
    return '\n'.join([
      "--------------------------------------------------",
      "Failed to generate grammar"
      ])
  
  a = GrammarAnalyzer(g)

  return '\n'.join([
    "--------------------------------------------------",
    renderGrammar(g),
    "",
    "Nullable: %s" % a.nullable,
    "First Sets: %s" % a.firstSets,
    "Follow Sets: %s" % a.followSets,
    "LL1: %s" % a.isLL1,
    "Parse Table: %s" % a.parseTable,
    renderParseTable(a.parseTable)
    ])
  

class TestGrammarClassifier(unittest.TestCase):
  
  def test_reachable1(self):
    a = GrammarAnalyzer({
      'A' : ['xyz', 'abBc'],
      'B' : ['asdf']
    })
    self.assertTrue(a.reachable('B','A'))
    self.assertFalse(a.reachable('A','B'))
    self.assertFalse(a.reachable('A','A'))
    self.assertFalse(a.reachable('B','B'))

  def test_reachable2(self):
    """

    Make sure we don't go into an infinite loop if the grammar is
    cyclic.

    """
    a = GrammarAnalyzer({
      'A' : ['xyz', 'abBc'],
      'B' : ['asdf', 'asdfBqwer']
    })
    self.assertTrue(a.reachable('B','A'))
    self.assertFalse(a.reachable('A','B'))
    self.assertFalse(a.reachable('A','A'))
    self.assertTrue(a.reachable('B','B'))

  def test_firstFollow(self):
    a = GrammarAnalyzer({
      'A' : ['xyz', 'abBc'],
      'B' : ['asdf', 'asdfBqwer']
    })
    self.assertEquals(a.firstSets['A'], set(['x','a']))
    self.assertEquals(a.firstSets['B'], set(['a']))
    self.assertEquals(a.followSets['A'], set(['$']))
    self.assertEquals(a.followSets['B'], set(['c','q']))

  def test_followLastCharIsNT(self):
    a = GrammarAnalyzer({
      'A' : ['xyz', 'abBc'],
      'B' : ['asdfA', 'AB']
    })
    self.assertEquals(a.firstSets['A'], set(['x','a']))
    self.assertEquals(a.firstSets['B'], set(['x','a']))
    self.assertEquals(a.followSets['A'], set(['$','x','a','c']))
    self.assertEquals(a.followSets['B'], set(['c']))

  def test_firstSet_constraints(self):
    a = GrammarAnalyzer({
      'A' : ['xyz', 'abBc'],
      'B' : ['asdfA', 'AB']
    })
    
    firsts = a.firstSet_constraints()
    
    self.assertFalse(a.firstConstraintCycles())
    self.assertEquals(a.firstSets['A'], firsts['A'])
    self.assertEquals(a.firstSets['B'], firsts['B'])

  def test_firstSet_constraints2(self):
    a = GrammarAnalyzer({
      'A' : ['xyz', 'Bc'],
      'B' : ['asdfA', 'AB']
    })
    
    firsts = a.firstSet_constraints()
    
    self.assertTrue(a.firstConstraintCycles())
    self.assertEquals(a.firstSets['A'], firsts['A'])
    self.assertEquals(a.firstSets['B'], firsts['B'])

  def test_followSet_constraints(self):
    a = GrammarAnalyzer({
      'A' : ['xyz', 'abBc'],
      'B' : ['asdfA', 'AB']
    })
    
    follows = a.followSet_constraints()
    print a.followConstraintCycles()
    
    self.assertEquals(a.followSets['A'], follows['A'])
    self.assertEquals(a.followSets['B'], follows['B'])

  def test_findSuperEasyNonLL1(self):
    g = findGrammar(
      lambda a: (not a.isLL1 and
                 a.countNullable() == 0 and
                 not a.followConstraintCycles() and
                 not a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findSuperEasyLL1(self):
    g = findGrammar(
      lambda a: (a.isLL1 and
                 a.countNullable() == 0 and
                 not a.followConstraintCycles() and
                 not a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findEasyNonLL1(self):
    g = findGrammar(
      lambda a: (not a.isLL1 and
                 not a.followConstraintCycles() and
                 not a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)
    
  def test_findEasyLL1(self):
    g = findGrammar(
      lambda a: (a.isLL1 and
                 not a.followConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findMed1NonLL1(self):
    g = findGrammar(
      lambda a: (not a.isLL1 and
                 not a.followConstraintCycles() and
                 a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findMed1LL1(self):
    g = findGrammar(
      lambda a: (a.isLL1 and
                 a.followConstraintCycles() and
                 True) #a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findMed2NonLL1(self):
    g = findGrammar(
      lambda a: (not a.isLL1 and
                 a.followConstraintCycles() and
                 not a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findMed2LL1(self):
    g = findGrammar(
        lambda a: (a.isLL1 and
                   a.followConstraintCycles() and
                   not a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)

  def test_findHardNonLL1(self):
    g = findGrammar(
      lambda a: (not a.isLL1 and
                 a.countNullable() > 0 and
                 a.followConstraintCycles() and
                 a.firstConstraintCycles())
    )
    self.assertIsNotNone(g)
    print grammarSpecs(g)


  def test_findHardLL1(self):
    g = findGrammar(
      lambda a: (a.isLL1 and
                 a.countNullable() > 0 and
                 #a.firstConstraintCycles() and 
                 a.followConstraintCycles() and
                 True)
      #gen=generateLL1Grammar,
      #maxProds=4,
      #nts="ABC"
    )
    self.assertIsNotNone(g)
    print "                HARD      ------------"
    print grammarSpecs(g)

  def testRandomParseTable(self):
    tblGen = ParseTableGenerator("ABC", "xyz", 3, 3)
    print renderGrammar(tblGen.g)
    print renderParseTable(tblGen.tbl)
    
if __name__ == "__main__":
  unittest.main()
