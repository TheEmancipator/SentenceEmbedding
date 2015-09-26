package kr.kaist.sentence.embedding.structure;

import java.util.Vector;

public class Tree {
	// variable declaration
	public Vector<Node> allNodes = new Vector<Node>();
	public Vector<Integer> leafNodeList = new Vector<Integer>();
	public Vector<Integer> wordList = new Vector<Integer>();

	public int dimension;

	public Tree(int dimension) {
		this.dimension=dimension;
	}
}