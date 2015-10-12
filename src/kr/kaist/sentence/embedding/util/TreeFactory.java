package kr.kaist.sentence.embedding.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

import kr.kaist.sentence.embedding.structure.Document;
import kr.kaist.sentence.embedding.structure.Node;
import kr.kaist.sentence.embedding.structure.Tree;

public class TreeFactory {
	public Vector<Integer> getOffspring(int nodeIndex, Tree tree) {
		//Get offsprings for a specific node
		Node node = tree.allNodes.get(nodeIndex);
		if(node.isLeaf) {
			// if the node is a leaf node(no child), add its index into offspring leave list
			node.offspringLeavesList.addElement(node.wordIndex);
			return node.offspringLeavesList;
		}
		if(node.childrenList.size() == 1) {
			// for the case that the node has only one child, get its offspring recursively
			int childIndex = node.childrenList.get(0);
			Vector<Integer> childOffspring = getOffspring(childIndex, tree);
			for(int i = 0; i < childOffspring.size(); i++)
				node.offspringLeavesList.addElement(childOffspring.get(i));
			return node.offspringLeavesList;
		} else {
			// for the case that the node has two children, get their offspring recursively
			int firstChildIndex = node.childrenList.get(0);
			int secondChildIndex = node.childrenList.get(1);
			Vector<Integer>firstChildOffspring = getOffspring(firstChildIndex, tree);
			Vector<Integer>secondChildOffspring = getOffspring(secondChildIndex, tree);
			for(int i = 0; i < firstChildOffspring.size(); i++) {
				int index = firstChildOffspring.get(i);
				if(node.offspringLeavesList.indexOf(index) == -1)
					// if the offspring of the first child is not in the offspring leave list, add it into the list
					node.offspringLeavesList.addElement(index);
			}
			for(int i = 0; i < secondChildOffspring.size(); i++) {
				int index = secondChildOffspring.get(i);
				if(node.offspringLeavesList.indexOf(index) == -1)
					// if the offspring of the second child is not in the offspring leave list, add it into the list
					node.offspringLeavesList.addElement(index);
			}
			return node.offspringLeavesList;
		}
	}

	public double[] getVector(int nodeIndex, double[][] weightMatrix, double[] bias, double[][] wordVectors, Tree tree){
		// get vector representation for an internal node
		// weightMatrix = d X 2d
		// bias = d
		Node node = tree.allNodes.get(nodeIndex);
		node.gradientVector = new double[tree.dimension];
		if(node.isLeaf) {
			// if the node is a leaf node(no child), add its word embedding and the gradient of the word embedding for tanh derivative 
			node.vector = copyVector(wordVectors[node.wordIndex]);
			node.gradientVector = tanhDerivativeVector(node.vector);
			tree.allNodes.set(nodeIndex, node);
			return node.vector;
		}
		if(node.childrenList.size() == 1) {
			// for the case that the node has only one child, get its offspring's word embedding and the gradient for tanh derivative recursively
			int childIndex = node.childrenList.get(0);
			node.vector = getVector(childIndex, weightMatrix, bias, wordVectors, tree);
			node.gradientVector = tanhDerivativeVector(node.vector);
			tree.allNodes.set(nodeIndex, node);
			return node.vector;
		} else {
			// for the case that the node has two children, get their offsprings' word embeddings and the gradients for tanh derivative recursively
			double[] firstChildVector = new double[tree.dimension];
			double[] secondChildVector = new double[tree.dimension];
			int firstChildIndex = node.childrenList.get(0);
			int secondChildIndex = node.childrenList.get(1);
			firstChildVector = getVector(firstChildIndex, weightMatrix, bias, wordVectors, tree);
			secondChildVector = getVector(secondChildIndex, weightMatrix, bias, wordVectors, tree);
			double[] concatenatedVector = new double[tree.dimension*2];
			// concatenate firstChildVector and secondChildVector
			for(int i = 0; i < tree.dimension; i++)
				concatenatedVector[i] = firstChildVector[i];
			for(int i=0; i < tree.dimension; i++)
				concatenatedVector[tree.dimension + i] = secondChildVector[i];
			node.vector = tanhVector(sumVectors(multiplyMatrixVector(weightMatrix, concatenatedVector), bias));
			node.gradientVector = tanhDerivativeVector(node.vector);
			tree.allNodes.set(nodeIndex, node);
			return node.vector;
		}
	}

	public void binarizeTree(Tree tree) {
		// transform the tree into a binary tree
        int allNodesSize = tree.allNodes.size();
        for(int i = (allNodesSize - 1); i > -1; i--) {
            Node node=tree.allNodes.get(i);
            if(node.childrenList.size() > 2) {
                int right = node.childrenList.get(node.childrenList.size() - 1);
                int finalRight = -1;
                int finalLeft = node.childrenList.get(0);
                for(int j = node.childrenList.size() - 2; j > -1; j--) {
                	int left = node.childrenList.get(j);
                    Node tempNode = new Node(tree.dimension);
                    tempNode.vector = new double[tree.dimension];
                    tempNode.childrenList.addElement(left);
                    tempNode.childrenList.addElement(right);
                    int parent = -1;
                    if(j == 0)
                        parent = i;
                    else 
                        parent = tree.allNodes.size() - 1;
                    tempNode.parent = parent;
                    tempNode.tag = "INTERN";
                    tempNode.index = tree.allNodes.size();
                    Node leftnode = tree.allNodes.get(left);
                    Node rightnode = tree.allNodes.get(right);
                    tree.allNodes.addElement(tempNode);
                    leftnode.parent = tree.allNodes.size() - 1;
                    rightnode.parent = tree.allNodes.size() - 1;
                    tree.allNodes.set(left, leftnode);
                    tree.allNodes.set(right, rightnode);
                    right = tree.allNodes.size()-1;
                    if(j == 0)
                    	finalRight = right;
                }
                node.childrenList = new Vector<Integer>();
                node.childrenList.addElement(finalLeft);
                node.childrenList.addElement(finalRight);
                tree.allNodes.set(i, node);
            }
		}
	}
	
	public void collapseUnaryTransformer(Tree tree) {
        int allNodesSize = tree.allNodes.size();
        for(int i = 0; i < allNodesSize; i++)	// initialize children list
        	tree.allNodes.get(i).childrenList = new Vector<Integer>();
        for(int i = 1; i < allNodesSize; i++)	// rewrite children list
        	tree.allNodes.get(tree.allNodes.get(i).parent).childrenList.addElement(tree.allNodes.get(i).index);
		for(int i = allNodesSize - 1; i > -1; i--) {
			Node node = tree.allNodes.get(i);
			if(node.childrenList.size() == 1) {
				for(int m = 0; m < allNodesSize; m++)
					if(tree.allNodes.get(m).index == node.parent)
						for(int j = 0; j < tree.allNodes.get(m).childrenList.size(); j++) {
							if(tree.allNodes.get(m).childrenList.get(j) == node.index)
								tree.allNodes.get(m).childrenList.set(j, tree.allNodes.get(node.childrenList.get(0)).index);
						}
				tree.allNodes.get(tree.allNodes.get(i).childrenList.get(0)).parent = tree.allNodes.get(i).parent;
				tree.allNodes.set(i, tree.allNodes.get(tree.allNodes.get(i).childrenList.get(0)));
			}
		}
		Vector<Node>newAllNodes = new Vector<Node>();
		for(Node node : tree.allNodes) {
			boolean redundancyChecker = false;
			for(Node newNode : newAllNodes)
				if(node.index == newNode.index)
					redundancyChecker = true;
			if(redundancyChecker == false)
				newAllNodes.addElement(node);
		}
		tree.allNodes = newAllNodes;
		
		Queue queue = new LinkedList();
		queue.add(tree.allNodes.get(0));
		Vector<Integer> indexMap = new Vector<Integer>();
		indexMap.addElement(tree.allNodes.get(0).index);
		while(!queue.isEmpty()) {
			Node node = (Node)queue.remove();
			for(int i = 0; i < node.childrenList.size(); i++) {
				for(int j = 0; j < tree.allNodes.size(); j++) {
					if(tree.allNodes.get(j).index == node.childrenList.get(i)) {
						queue.add(tree.allNodes.get(j));
						indexMap.addElement(node.childrenList.get(i));
					}
				}
			}
		}
		
		for(int i = 0; i < tree.allNodes.size(); i++) {
			for(int m = 0; m < indexMap.size(); m++) {
				if(indexMap.get(m) == tree.allNodes.get(i).index) {
					tree.allNodes.get(i).index = m;
					break;
				}
			}
		}

		for(int i = 0; i < tree.allNodes.size(); i++) {
			for(int m = 0; m < indexMap.size(); m++) {
				if(indexMap.get(m) == tree.allNodes.get(i).parent) {
					tree.allNodes.get(i).parent = m;
					break;
				}
			}
		}
		
		for(int i = 0; i < tree.allNodes.size(); i++) {
			for(int j = 0; j < tree.allNodes.get(i).childrenList.size(); j++) {
				for(int m = 0; m < indexMap.size(); m++) {
					if(indexMap.get(m) == tree.allNodes.get(i).childrenList.get(j)) {
						tree.allNodes.get(i).childrenList.set(j, m);
						break;
					}
				}
			}
		}
		
		newAllNodes = new Vector<Node>();
		for(int i = 0; i < tree.allNodes.size(); i++)
			for(int j = 0; j < tree.allNodes.size(); j++)
				if(tree.allNodes.get(j).index == i) {
					newAllNodes.addElement(tree.allNodes.get(j));
					break;
				}
		tree.allNodes = newAllNodes;
		
	}

	
	public void readTree(String inputText, HashMap<String,Integer>WordToNum, Tree tree) {
		//Read a parse tree
		int i = 0;
		Stack<Integer>stack =new Stack<Integer>();
		while(i < inputText.length()) {
			if(inputText.charAt(i) == '(') {
				String posTag = "";
				i++;
				while((inputText.charAt(i) != '(') && (inputText.charAt(i) != ')')) {
					posTag += inputText.charAt(i);
					i++;
				}
				posTag = posTag.trim();
				Node node = new Node(tree.dimension);
				node.index = tree.allNodes.size();
				node.tag = posTag;
				if(!stack.isEmpty()) {
					Node rootNode = tree.allNodes.get(stack.peek());
					rootNode.childrenList.addElement(node.index);
					node.parent = rootNode.index;
					node.vector = new double[tree.dimension];
					tree.allNodes.set(rootNode.index, rootNode);
				}
				tree.allNodes.addElement(node);
				stack.push(node.index);
			} else if(inputText.charAt(i)==')') {
				Node rootNode = tree.allNodes.get(stack.peek());
				String posWord = rootNode.tag;
				if(posWord.indexOf(" ") != -1) {
					int delimeterIndex = posWord.indexOf(" ");
					String posTag = posWord.substring(0, delimeterIndex);
					String word = posWord.substring(delimeterIndex + 1);
					rootNode.tag = posTag;
					rootNode.wordIndex = WordToNum.get(word.toLowerCase());
					rootNode.word = word;
					rootNode.isLeaf = true;
					tree.leafNodeList.addElement(rootNode.index);
					tree.allNodes.set(rootNode.index, rootNode);
					if(tree.wordList.indexOf(rootNode.wordIndex) == -1)
						tree.wordList.addElement(rootNode.wordIndex);
				}
				stack.pop();
				i++;
			}
			else if(inputText.charAt(i)==' ')i++;
		}
	}
	
	public double[] sumVectors(double[] inputVector1, double[] inputVector2) {
		// get sum of two input vectors
		if(inputVector1.length != inputVector2.length) {
			System.out.println("lengths of the inputs are different [in sumVector].\nProcess has been terminated.");
			System.exit(-1);
		}
		double[] sumVector = new double[inputVector1.length];
		for(int i = 0; i < inputVector1.length; i++)
			sumVector[i] = inputVector1[i] + inputVector1[i];
		return sumVector;
	}

	public double[] tanhVector(double[] inputVector) {
		// get tanh of the input vector
		double[] tanhVector = new double[inputVector.length];
		for(int i = 0; i < inputVector.length; i++) {
			tanhVector[i] = (Math.exp(inputVector[i])-Math.exp(-inputVector[i]))/(Math.exp(inputVector[i])+Math.exp(-inputVector[i]));
		}
		return tanhVector;
	}

	public double[] tanhDerivativeVector(double[] inputVector) {
		// get gradient of the input vector for tanh derivative
		double[] gradientVector;
		gradientVector = new double[inputVector.length];
		for(int i = 0; i < inputVector.length; i++)
			gradientVector[i] = 1 - Math.pow(inputVector[i], 2);
		return gradientVector;
	}

	public double[] copyVector(double[] inputVector) {
		// make a copy for vector
		double[] copiedVector;
		copiedVector =new double[inputVector.length];
		for(int i = 0; i < inputVector.length; i++)
			copiedVector[i] = inputVector[i];
		return copiedVector;
	}

	public double[] multiplyMatrixVector(double[][] inputMatrix, double[] inputVector){
		// get multiplication of the input Matrix and the input vector 
		if(inputMatrix[0].length != inputVector.length) {
			System.out.println("lengths of the inputs are different [in multiplyMatrixVector].\nProcess has been terminated.");
			System.exit(-1);
		}  		
		double[] multiplication = new double[inputMatrix.length];
		for(int i = 0; i < inputMatrix.length; i++){
			multiplication[i] = 0;
			for(int j = 0; j < inputMatrix[0].length; j++)
				multiplication[i] += inputMatrix[i][j] * inputVector[j];
		}
		return multiplication;
	}

	/*public void makeTestSet() {
		Node node0 = new Node(dimension);
		node0.vector = new double[dimension];
		node0.use = new double[dimension];
		node0.parent = -1;
		node0.index = 0;
		node0.childrenList = new Vector<Integer>();
		node0.childrenList.addElement(1);
		node0.childrenList.addElement(2);
		node0.childrenList.addElement(3);

		Node node1 = new Node(dimension);
		node1.vector = new double[dimension];
		node1.use = new double[dimension];
		node1.parent = 0;
		node1.index = 1;
		node1.isLeaf = true;

		Node node2 = new Node(dimension);
		node2.vector = new double[dimension];
		node2.use = new double[dimension];
		node2.parent = 0;
		node2.index = 2;
		node2.childrenList.addElement(4);
		node2.childrenList.addElement(5);
		node2.childrenList.addElement(6);

		Node node3 = new Node(dimension);
		node3.vector = new double[dimension];
		node3.use = new double[dimension];
		node3.parent = 0;
		node3.index = 3;
		node3.isLeaf = true;

		Node node4 = new Node(dimension);
		node4.vector = new double[dimension];
		node4.use = new double[dimension];
		node4.parent = 2;
		node4.index = 4;
		node4.isLeaf = true;

		Node node5 = new Node(dimension);
		node5.vector = new double[dimension];
		node5.use = new double[dimension];
		node5.parent = 2;
		node5.index = 5;
		node5.isLeaf = true;

		Node node6 = new Node(dimension);
		node6.vector = new double[dimension];
		node6.use = new double[dimension];
		node6.parent = 2;
		node6.index = 6;
		node6.isLeaf = true;

		allNodes.addElement(node0);
		allNodes.addElement(node1);
		allNodes.addElement(node2);
		allNodes.addElement(node3);
		allNodes.addElement(node4);
		allNodes.addElement(node5);
		allNodes.addElement(node6);
	}*/
	

	/*public void print(){
		for(int i = 0; i < allNodes.size(); i++){
			Node node = allNodes.get(i);
			System.out.println(i+" "+node.parent+" "+node.childrenList+" "+node.word+" "+node.isLeaf+" "+node.wordIndex+" "+node.offspringLeavesList);
		}
	}*/
	
	/*
	 * for backup
	 	public void binarizeTree(Tree tree) {
		// transform the tree into a binary tree
		for(int i = 0; i < tree.allNodes.size(); i++) {
			Node node = tree.allNodes.get(i);
			if(node.childrenList.size() > 2) {
				// if the number of children exceeds 2, make it 2 by adding an internal node which has the right-most two nodes as children 
				int finalLeftChildIndex = node.childrenList.get(0);

				// create a new internal node just right to the left-first child node
				int tempNodeIndex = finalLeftChildIndex + 1;
				Node tempNode = new Node(tree.dimension);
				tempNode.vector = new double[tree.dimension];
				tempNode.parent = i;
				tempNode.posTag = "INTERN";
				tempNode.index = tempNodeIndex;

				// migrate all the children nodes, except the left-first child node, into the children list of the new node 
				for(int j = 1; j < node.childrenList.size(); j++) {
					tree.allNodes.get(node.childrenList.get(j)).parent = tempNodeIndex;
					tempNode.childrenList.addElement(node.childrenList.get(j) + 1); // add 1 here because the nodes' index will be updated soon
				}
				// add new internal node into the node vector
				tree.allNodes.add(tempNodeIndex, tempNode);

				// because an element(the new internal node) has been added, update indexes (just +1)
				for(int j = tempNodeIndex+1; j < tree.allNodes.size(); j++) {
					// update node index 
					int previousIndex = tree.allNodes.get(j).index;
					tree.allNodes.get(j).index = previousIndex + 1;

					// update children list
					if(tree.allNodes.get(j).childrenList.size() > 0) {
						for(int m = 0; m < tree.allNodes.get(j).childrenList.size(); m++) {
							int previousChildIndex = tree.allNodes.get(j).childrenList.get(m);
							tree.allNodes.get(j).childrenList.set(m, previousChildIndex + 1);
						}
					}
				}

				// modify root's children list
				node.childrenList = new Vector<Integer>();
				node.childrenList.addElement(finalLeftChildIndex);
				node.childrenList.addElement(tempNodeIndex);
				tree.allNodes.set(i, node);
			}
		}
	}
	 */
}