package kr.kaist.sentence.embedding.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Stack;
import java.util.Arrays;
import java.util.Vector;

import kr.kaist.sentence.embedding.structure.Document;
import kr.kaist.sentence.embedding.structure.Node;
import kr.kaist.sentence.embedding.structure.Tree;

public class Test {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String inputFileName = "data/neg_tree/cv012_29411.txt.tree";
		Document document = new Document(50);
		readRSTTree(inputFileName, document);
		setRSTUnitToSentence(document);
		for(int i = 0 ; i < document.leafNodeList.size(); i++)
			System.out.print(document.leafNodeList.get(i) + " ");
		System.out.println("");	
		for(int i = 0 ; i < document.allNodes.size(); i++)
			if(document.allNodes.get(i).isLeaf)
				System.out.print(document.allNodes.get(i).word + "\n");
		System.out.println("");
		alignRSTTree(document);
		
		System.out.println(document.allNodes.size());

		for(int i = 0 ; i < document.leafNodeList.size(); i++)
			System.out.print(document.leafNodeList.get(i) + " ");
		System.out.println("");
		for(int i = 0 ; i < document.allNodes.size(); i++)
			if(document.allNodes.get(i).isLeaf)
				System.out.print(document.allNodes.get(i).index + " ");
		System.out.println("");
		for(int i = 0 ; i < document.allNodes.size(); i++)
			System.out.print(document.allNodes.get(i).parent + " ");
	}
	
	public static void alignRSTTree(Document document) {
	       int allNodesSize = document.allNodes.size();
	       for(int i = 0; i < allNodesSize; i++)	// initialize children list
	    	   document.allNodes.get(i).childrenList = new Vector<Integer>();
	       	for(int i = 1; i < allNodesSize; i++)	// rewrite children list
	        	document.allNodes.get(document.allNodes.get(i).parent).childrenList.addElement(document.allNodes.get(i).index);
	       	for(int i = 0; i < allNodesSize; i++) {
				if(document.allNodes.get(i).isLeaf == true) {
					if(document.allNodes.get(document.allNodes.get(i).parent).childrenList.size() == 1) {
						if(document.allNodes.get(document.allNodes.get(document.allNodes.get(i).parent).parent).childrenList.size() == 1) {
							int parentIndex = document.allNodes.get(i).parent;
							int grandParentIndex = document.allNodes.get(document.allNodes.get(i).parent).parent; 
							document.allNodes.get(i).parent = document.allNodes.get(grandParentIndex).parent;
							document.allNodes.get(i).index = grandParentIndex;
							document.allNodes.set(parentIndex, document.allNodes.get(i));
							document.allNodes.set(grandParentIndex, document.allNodes.get(i));
						} else {
							int parentIndex = document.allNodes.get(i).parent; 
							document.allNodes.get(i).parent = document.allNodes.get(parentIndex).parent;
							document.allNodes.get(i).index = parentIndex;
							document.allNodes.set(parentIndex, document.allNodes.get(i));							
						}
					}
				}
			}
			Vector<Node>newAllNodes = new Vector<Node>();
			for(Node node : document.allNodes) {
				boolean redundancyChecker = false;
				for(Node newNode : newAllNodes)
					if(node.index == newNode.index)
						redundancyChecker = true;
				if(redundancyChecker == false)
					newAllNodes.addElement(node);
			}
			document.allNodes = newAllNodes;
			
			Vector<Integer> vect = new Vector<Integer>();
			for(int i = 0; i < document.allNodes.size(); i++) {
				vect.addElement(document.allNodes.get(i).index);
			}
			
			Object[] indexMap = vect.toArray();
			Arrays.sort(indexMap);
			
			for(int i = 0; i < document.allNodes.size(); i++) {
				for(int m = 0; m < indexMap.length; m++) {
					if(Integer.parseInt(indexMap[m].toString()) == document.allNodes.get(i).index) {
						document.allNodes.get(i).index = m;
						break;
					}
				}
			}

			for(int i = 0; i < document.allNodes.size(); i++) {
				for(int m = 0; m < indexMap.length; m++) {
					if(Integer.parseInt(indexMap[m].toString()) == document.allNodes.get(i).parent) {
						document.allNodes.get(i).parent = m;
						break;
					}
				}
			}
			
			for(int i = 0; i < document.allNodes.size(); i++) {
				if(document.allNodes.get(i).isLeaf == false) {
					for(int m = 0; m < indexMap.length; m++) 
						if(Integer.parseInt(indexMap[m].toString()) == document.allNodes.get(i).childrenList.get(0))
							document.allNodes.get(i).childrenList.set(0, m);
					for(int m = 0; m < indexMap.length; m++) 
						if(Integer.parseInt(indexMap[m].toString()) == document.allNodes.get(i).childrenList.get(1))
							document.allNodes.get(i).childrenList.set(1, m);
				}
			}
			
			newAllNodes = new Vector<Node>();
			for(int i = 0; i < document.allNodes.size(); i++)
				for(int j = 0; j < document.allNodes.size(); j++)
					if(document.allNodes.get(j).index == i) {
						newAllNodes.addElement(document.allNodes.get(j));
						break;
					}
			
			document.allNodes = newAllNodes;
			
			Vector<Integer> newLeafNodeList = new Vector<Integer>();
			for(int i = 0; i < document.allNodes.size(); i++) {
				if(document.allNodes.get(i).isLeaf == true)
					newLeafNodeList.addElement(document.allNodes.get(i).index);
			}
			document.leafNodeList = newLeafNodeList;
			
			
	}
	
	public static void setRSTUnitToSentence(Document document) {
		String sentence = "";
		int superParentIndex = 9999;
		boolean pendingMode = false;
		
		Vector newleafNodeList = new Vector<Integer>();

		for(int i = 0; i < document.leafNodeList.size(); i++) {
			Node node = document.allNodes.elementAt(document.leafNodeList.elementAt(i));
			if(!(node.word.endsWith("<s>") || node.word.endsWith("<p>")) && pendingMode == false) {
				if(superParentIndex > node.parent)
					superParentIndex = node.parent;
				sentence = sentence + " " + node.word;
				//document.allNodes.remove(document.leafNodeList.elementAt(i));
				//document.leafNodeList.remove(i);
				pendingMode = true;
			} else if(!(node.word.endsWith("<s>") || node.word.endsWith("<p>")) && pendingMode == true){
				if(superParentIndex > node.parent)
					superParentIndex = node.parent;
				sentence = sentence + " " + node.word;
				//document.allNodes.remove(document.leafNodeList.elementAt(i));
				//document.leafNodeList.remove(i);
			} else if((node.word.endsWith("<s>") || node.word.endsWith("<p>")) && pendingMode == true){
				if(superParentIndex > node.parent)
					superParentIndex = node.parent;
				sentence = sentence + " " + node.word;
				//document.allNodes.remove(document.leafNodeList.elementAt(i));
				//document.leafNodeList.remove(i);
				//System.out.println(document.allNodes.elementAt(superParentIndex).word);
				//System.out.println(superParentIndex);
				Node newNode = new Node(document.dimension);
				Node leftChildNode = new Node(document.dimension);
				for(int j = 0; j < document.allNodes.size(); j++)
					if(document.allNodes.elementAt(j).index == superParentIndex) {
						//System.out.println(document.allNodes.elementAt(superParentIndex).tag);
						//System.out.println(document.allNodes.elementAt(document.allNodes.elementAt(j).childrenList.get(0)).word);
						//System.out.println(document.allNodes.elementAt(document.allNodes.elementAt(j).childrenList.get(0)).tag);
						//System.out.println(node.word + pendingMode);
						leftChildNode = document.allNodes.elementAt(document.allNodes.elementAt(j).childrenList.get(0));
					}
				if(leftChildNode.word.endsWith("<s>") || leftChildNode.word.endsWith("<p>")) {
					newNode.index = document.allNodes.elementAt(superParentIndex).childrenList.elementAt(1);
					newNode.parent = superParentIndex;
					Node parent = document.allNodes.get(superParentIndex);
					parent.childrenList.set(1, newNode.index);
					document.allNodes.set(superParentIndex, parent);
				} else {
					newNode.index = superParentIndex;
					newNode.parent = document.allNodes.elementAt(newNode.index).parent;					
				}
				newNode.isLeaf = true;
				sentence = sentence.replaceAll("<s>", "");
				sentence = sentence.replaceAll("<p>", "");
				newNode.word = sentence.trim();
				newNode.vector = new double[document.dimension];				
				document.allNodes.set(newNode.index, newNode);
				newleafNodeList.addElement(newNode.index);
				
				superParentIndex = 9999;
				sentence = "";
				pendingMode = false;
			} else if((node.word.endsWith("<s>") || node.word.endsWith("<p>")) && pendingMode == false) {
				node.word = node.word.replaceAll("<s>", "");
				node.word = node.word.replaceAll("<p>", "");
				node.word = node.word.trim();
				document.allNodes.set(node.index, node);
				newleafNodeList.addElement(node.index);
				//System.out.println(node.word + pendingMode);
			} else
				;
		}
		document.leafNodeList = newleafNodeList;
		Vector<Node> newAllNodes = new Vector<Node>();
		for(int i = 0; i < document.leafNodeList.size(); i++) {
			Node node = document.allNodes.get(document.leafNodeList.get(i));
			newAllNodes.addElement(node);
			Node parentNode = document.allNodes.get(node.parent);
			while(parentNode.parent != -1) {
				boolean redundancyChecker = false;
				for(Node newNode : newAllNodes)
					if(parentNode.index == newNode.index) {
						redundancyChecker = true;
						break;
					}
				if(redundancyChecker == false)
					newAllNodes.addElement(parentNode);
				parentNode = document.allNodes.get(parentNode.parent);
			}
		}
		newAllNodes.addElement(document.allNodes.get(0));
		document.allNodes = newAllNodes;
		
		Vector<Integer> vect = new Vector<Integer>();
		for(int i = 0; i < document.allNodes.size(); i++) {
			vect.addElement(document.allNodes.get(i).index);
		}

		Object[] indexMap = vect.toArray();
		Arrays.sort(indexMap);
		
		for(int i = 0; i < document.leafNodeList.size(); i++) {
			for(int m = 0; m < indexMap.length; m++) {
				if(Integer.parseInt(indexMap[m].toString()) == document.leafNodeList.get(i)) {
					document.leafNodeList.set(i, m);
					break;
				}
			}
		}
		
		for(int i = 0; i < document.allNodes.size(); i++) {
			for(int m = 0; m < indexMap.length; m++) {
				if(Integer.parseInt(indexMap[m].toString()) == document.allNodes.get(i).index) {
					document.allNodes.get(i).index = m;
					break;
				}
			}
		}

		for(int i = 0; i < document.allNodes.size(); i++) {
			for(int m = 0; m < indexMap.length; m++) {
				if(Integer.parseInt(indexMap[m].toString()) == document.allNodes.get(i).parent) {
					document.allNodes.get(i).parent = m;
					break;
				}
			}
		}		
		
		newAllNodes = new Vector<Node>();
		for(int i = 0; i < document.allNodes.size(); i++)
			for(int j = 0; j < document.allNodes.size(); j++)
				if(document.allNodes.get(j).index == i) {
					newAllNodes.addElement(document.allNodes.get(j));
					break;
				}
		
		document.allNodes = newAllNodes;
	}
	
	public static void readRSTTree(String inputFileName, Document document) throws IOException {
		//Read a RST parse tree
		
		BufferedReader inFile = new BufferedReader(new FileReader(inputFileName));
		String inputText = "";
		String sLine = "";
		while((sLine = inFile.readLine()) != null) {
			if(sLine.trim().equals("") || sLine.trim().equals("\n"))
				;
			else
				inputText += sLine + "\n";
		}
		
		int i = 0;
		boolean textSpanMode = false;
		Stack<Integer>stack =new Stack<Integer>();
		
		while(i < inputText.length()) {
			// handling RST tag
			if(inputText.charAt(i) == '(' && textSpanMode == false) {
				String RSTTag = "";
				i++;
				while((inputText.charAt(i) != '[')) {
					RSTTag += inputText.charAt(i);
					i++;
				}
				RSTTag = RSTTag.trim();
				// (인 case나 -!인 case가 있다.
				// () (), _! (), () _!, _! _! 4개의 경우가 있음
				Node node = new Node(document.dimension);
				node.index = document.allNodes.size();
				node.tag = RSTTag;
				node.leftChildRole = inputText.charAt(i+1);
				node.rightChildRole = inputText.charAt(i+4);
				i = i + 6;
				if(!stack.isEmpty()) {
					Node rootNode = document.allNodes.get(stack.peek());
					rootNode.childrenList.addElement(node.index);
					node.parent = rootNode.index;
					node.vector = new double[document.dimension];
					document.allNodes.set(rootNode.index, rootNode);
				}
				document.allNodes.addElement(node);
				stack.push(node.index);
			// handling text span
			} else if(inputText.charAt(i) == '_' && inputText.charAt(i+1) == '!' && textSpanMode == false) {
				i = i + 2;
				textSpanMode = true;
				String textSpan = "";
				
				while(!(inputText.charAt(i) == '!' && inputText.charAt(i+1) == '_')) {
					textSpan += inputText.charAt(i);
					i++;
				}
				Node node = new Node(document.dimension);
				node.index = document.allNodes.size();
				node.isLeaf = true;
				node.word = textSpan;
				document.leafNodeList.addElement(node.index);
				i = i + 2;

				if(!stack.isEmpty()) {
					Node rootNode = document.allNodes.get(stack.peek());
					rootNode.childrenList.addElement(node.index);
					node.parent = rootNode.index;
					node.vector = new double[document.dimension];
					document.allNodes.set(rootNode.index, rootNode);
				}
				document.allNodes.addElement(node);
				textSpanMode = false;
				
			} else if(inputText.charAt(i)==')' && textSpanMode == false) {
				//int index = stack.pop(); 
				//System.out.println(document.allNodes.elementAt(index).index + " " + document.allNodes.elementAt(index).tag+ " " + document.allNodes.elementAt(index).leftChildRole+ " " + document.allNodes.elementAt(index).rightChildRole);
				stack.pop();
				i++;
			} else {
				i++;
			}
		}
		
		document.allNodes.elementAt(0).parent = -1;
	}
}
