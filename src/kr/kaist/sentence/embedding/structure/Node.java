package kr.kaist.sentence.embedding.structure;

import java.util.Vector;

public class Node {
	public String word;
    public int wordIndex = -1;
    public String posTag;
    // public int parseIndex;
    public boolean isLeaf = false;
    public Vector<Integer> childrenList = new Vector<Integer>();
    public Vector<Integer> offspringLeavesList = new Vector<Integer>();
    public int parent = -1;
    // public int leftSibling = -1;
    // public int rightSibling = -1;
    public int index;
    public int dimension;
    public double[] vector;
    public double[] gradientVector;	// gradient = differentiated
    public double[] calculatedVector;
    public Node(int dimension) {
        this.dimension = dimension;
    }
}
