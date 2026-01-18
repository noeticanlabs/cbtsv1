import unittest

import noetica_nsc_phase1.nsc as nsc
from noetica_nsc_phase1.nsc_parser import Atom, Group, Phrase, Sentence, Program

class TestParseGroupsArrow(unittest.TestCase):
    def test_program_structure(self):
        src = "[φ↻]⇒[∆◯]□"
        prog, flat, bc, tpl = nsc.nsc_to_pde(src)
        # Assert program has one sentence
        self.assertEqual(len(prog.sentences), 1)
        sentence = prog.sentences[0]
        # Assert arrow is True
        self.assertTrue(sentence.arrow)
        # Assert lhs has one item: a Group
        self.assertEqual(len(sentence.lhs.items), 1)
        lhs_group = sentence.lhs.items[0]
        self.assertIsInstance(lhs_group, Group)
        self.assertEqual(lhs_group.delim, '[]')
        self.assertEqual(len(lhs_group.inner.items), 2)
        self.assertEqual(lhs_group.inner.items[0].value, 'φ')
        self.assertEqual(lhs_group.inner.items[1].value, '↻')
        # Assert rhs has two items: Group and Atom
        self.assertEqual(len(sentence.rhs.items), 2)
        rhs_group = sentence.rhs.items[0]
        self.assertIsInstance(rhs_group, Group)
        self.assertEqual(rhs_group.delim, '[]')
        self.assertEqual(len(rhs_group.inner.items), 2)
        self.assertEqual(rhs_group.inner.items[0].value, '∆')
        self.assertEqual(rhs_group.inner.items[1].value, '◯')
        rhs_atom = sentence.rhs.items[1]
        self.assertIsInstance(rhs_atom, Atom)
        self.assertEqual(rhs_atom.value, '□')