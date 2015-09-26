package kr.kaist.sentence.embedding.structure;

import java.util.Vector;

public class Document {
    public boolean tag; 
	public Vector<Integer> treeList = new Vector<Integer>();
	
	public int dimension;
	
	public Document(int dimension) {
		this.dimension=dimension;
	}
}
