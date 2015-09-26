package kr.kaist.sentence.embedding.util;

import java.io.IOException;

import kr.kaist.sentence.embedding.structure.Tree;

public class Test {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		ReadData test = new ReadData(50);
		
		test.readWordVector("data/all_vocabulary.txt", "data/all_wordvector.txt");
		test.read("data/test");
		
		
		/*
		//test.makeTestSet();
		System.out.println("==original==");
		for(int i = 0; i < test.allNodes.size(); i++) {
			System.out.print("node " + test.allNodes.get(i).index + " - children: ");
			for(int j = 0; j < test.allNodes.get(i).childrenList.size(); j++) {
				System.out.print(test.allNodes.get(i).childrenList.get(j) + ", ");
			}
			System.out.print("\n");
		}
		test.binarizeTree();
		System.out.println("==binarized==");
		for(int i = 0; i < test.allNodes.size(); i++) {
			System.out.print("node " + test.allNodes.get(i).index + " - children: ");
			for(int j = 0; j < test.allNodes.get(i).childrenList.size(); j++) {
				System.out.print(test.allNodes.get(i).childrenList.get(j) + ", ");
			}
			System.out.print("\n");
		}	
		*/	
	}
}
