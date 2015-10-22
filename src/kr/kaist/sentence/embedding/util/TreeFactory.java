package kr.kaist.sentence.embedding.util;

import java.util.*;

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

	public double[] getTreeVector(int nodeIndex, double[][] weightMatrix, double[] bias, double[][] wordVectors, Tree tree){
		// get vector representation for an internal node
		// weightMatrix = d X 2d
		// bias = d
		Node node = tree.allNodes.get(nodeIndex);
		node.vectorDerivative = new double[tree.dimension];
		if(node.isLeaf) {
			// if the node is a leaf node(no child), add its word embedding and the gradient of the word embedding for tanh derivative 
			node.vector = copyVector(wordVectors[node.wordIndex]);
			node.vectorDerivative = tanhDerivativeVector(node.vector);
			tree.allNodes.set(nodeIndex, node);
			return node.vector;
		}
		if(node.childrenList.size() == 1) {
			// for the case that the node has only one child, get its offspring's word embedding and the gradient for tanh derivative recursively
			int childIndex = node.childrenList.get(0);
			node.vector = getTreeVector(childIndex, weightMatrix, bias, wordVectors, tree);
			node.vectorDerivative = tanhDerivativeVector(node.vector);
			tree.allNodes.set(nodeIndex, node);
			return node.vector;
		} else {
			// for the case that the node has two children, get their offsprings' word embeddings and the gradients for tanh derivative recursively
			double[] firstChildVector = new double[tree.dimension];
			double[] secondChildVector = new double[tree.dimension];
			int firstChildIndex = node.childrenList.get(0);
			int secondChildIndex = node.childrenList.get(1);
			firstChildVector = getTreeVector(firstChildIndex, weightMatrix, bias, wordVectors, tree);
			secondChildVector = getTreeVector(secondChildIndex, weightMatrix, bias, wordVectors, tree);
			double[] concatenatedVector = new double[tree.dimension*2];
			// concatenate firstChildVector and secondChildVector
			for(int i = 0; i < tree.dimension; i++)
				concatenatedVector[i] = firstChildVector[i];
			for(int i=0; i < tree.dimension; i++)
				concatenatedVector[tree.dimension + i] = secondChildVector[i];
			node.vector = tanhVector(sumVectors(multiplyMatrixVector(weightMatrix, concatenatedVector), bias));
			node.vectorDerivative = tanhDerivativeVector(node.vector);
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
}