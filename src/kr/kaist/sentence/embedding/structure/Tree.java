package kr.kaist.sentence.embedding.structure;

import java.util.Vector;

public class Tree {
	// variable declaration
	public Vector<Node> allNodes;
	public Vector<Integer> leafNodeList;
	public Vector<Integer> wordList;

	public int dimension;

	public Tree(int dimension) {
		this.dimension=dimension;
		allNodes = new Vector<Node>();
		leafNodeList = new Vector<Integer>();
		wordList = new Vector<Integer>();
	}
}