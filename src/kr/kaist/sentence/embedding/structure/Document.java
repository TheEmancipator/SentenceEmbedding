package kr.kaist.sentence.embedding.structure;

import java.util.Vector;

public class Document {
    public int tag;
    public double inferredClass;
	public int dimension;
	public String fileName = "";
	
	//double join[];
	//double hiddenLayer[];
	//double derivativeHiddenLayer[];
	//double hDocument;
	
	public Vector<Node> allNodes;
	
	public Vector<Integer> treeList;
	public Vector<Integer> leafNodeList;
	
	public Document(int dimension) {
		this.dimension=dimension;
		
		allNodes = new Vector<Node>();
		treeList = new Vector<Integer>();
		leafNodeList = new Vector<Integer>();
	}
}