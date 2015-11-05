package kr.kaist.sentence.embedding.structure;

import java.util.Vector;

public class Tree {
	// common
	public Vector<Node> allNodes;
	public Vector<Integer> leafNodeList;
	public int dimension;
	
	// sentence
	public Vector<Integer> wordList;
	
	// Document 
    public int tag;
    public double inferredClass;
	public String fileName = "";

	public Tree(int dimension) {
		this.dimension=dimension;
		allNodes = new Vector<Node>();
		leafNodeList = new Vector<Integer>();
		wordList = new Vector<Integer>();
	}
}