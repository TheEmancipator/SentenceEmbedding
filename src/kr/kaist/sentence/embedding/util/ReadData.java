package kr.kaist.sentence.embedding.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Random;
import java.util.Vector;

import kr.kaist.sentence.embedding.structure.Batch;
import kr.kaist.sentence.embedding.structure.Document;
import kr.kaist.sentence.embedding.structure.Tree;

public class ReadData {
	
    public static double[][] wordVector;
    public static double[][] weightMatrix;
    public static double[] bias;
    //static HashMap<String,double[]>G=new HashMap<String,double[]>();
       
    public static Vector<Tree>allTree = new Vector<Tree>();
    public static Vector<Document>allDocument = new Vector<Document>();
    public static Vector<Batch>allBatch = new Vector<Batch>();
    //public static Vector<Batch>AllBatch=new Vector<Batch>();
    //public static Vector<Clique>AllClique=new Vector<Clique>();
    
	private static HashMap<String,Integer>WordToNum = new HashMap<String,Integer>();
	private static HashMap<Integer,String>NumToWord = new HashMap<Integer,String>();
    
    private static int dimension;
    
    
    
    public ReadData(int dimension){
        this.dimension = dimension;
    }
    
    public static void init() {
        weightMatrix = new double[dimension][dimension * 2];
        bias = new double[dimension];
        
        double epso=Math.sqrt(6)/(Math.sqrt(2 * dimension));
        Random r = new Random();
        
        for(int i = 0; i < dimension; i++) {
            bias[i]=r.nextDouble() * 2 * epso - epso;
            for(int j = 0; j < dimension * 2; j++){
                weightMatrix[i][j]=r.nextDouble() * 2 * epso - epso;
            }
        }
    }

    
    public void getBatch(){ 
        //minibatch
        int batchSize = 10;	// specify batch size
        int totalIteration = allDocument.size() / batchSize + 1;
        for(int i = 0; i < totalIteration; i++) {
            Batch batch=new Batch();
            int begin = batchSize * i;
            int end;
            if(i != (totalIteration - 1)) 
            	end = batchSize * (i + 1);
            else 
            	end = allDocument.size();
            for(int j = begin; j < end; j++) {
                Document document = allDocument.get(j);
                batch.documentList.addElement(j);
                for(int k = 0; k < document.treeList.size(); k++) {
                	batch.treeList.addElement(document.treeList.get(k));
                    for(int wordIndex : allTree.get(document.treeList.get(k)).wordList)
                    	if(batch.wordList.indexOf(wordIndex) == -1)
                    		batch.wordList.addElement(wordIndex);
                }
            }
            allBatch.addElement(batch);
        }
    }
    
    public static void read(String folderName) throws IOException{
        // read documents
        File folder = new File(folderName);
        File[] listofFiles = folder.listFiles();
        init();
        for(int i = 0; i < listofFiles.length; i++){
            String fileName = folderName + "/" + listofFiles[i].getName();
            Document doc = new Document(dimension);
            BufferedReader inFile = new BufferedReader(new FileReader(fileName));
            System.out.println(fileName);
            for(String line = inFile.readLine(); line != null; line = inFile.readLine()){
                Tree tree = new Tree(dimension);
                TreeFactory treeFactory = new TreeFactory(); 
                treeFactory.readTree(line, WordToNum, tree); // read a parse tree 
              
                treeFactory.binarizeTree(tree); // transform to binary tree
                
                treeFactory.collapseUnaryTransformer(tree);	// transform to collapse unary form
                treeFactory.getVector(0, weightMatrix, bias, wordVector, tree);	// calculate with recursive neural network scheme
                doc.treeList.addElement(allTree.size());	// save the index of trees in the document into document object 
                allTree.addElement(tree);
                /*
                // sentence checker
                System.out.println(fileName);
                for(int j = 0; j < tree.allNodes.size(); j++) {
                	if(tree.allNodes.get(j).isLeaf)
                		System.out.print(tree.allNodes.get(j).word + " ");
                }
                System.out.print("\n");
                */
            }
            allDocument.addElement(doc); // save document
        }
    }

    public static void readWordVector(String wordFile, String vectorFile)throws IOException { 
        // Read word embeddings (from Senna) from vector file
        System.out.println(wordFile + " " + vectorFile);
        BufferedReader inWordFile = new BufferedReader(new FileReader(wordFile));
        int numWordFileLines = 0;
        String wordFileLine, vectorFileLine;
        for(wordFileLine = inWordFile.readLine(); wordFileLine != null; wordFileLine = inWordFile.readLine()) {
            numWordFileLines++;
        }
        wordVector = new double[numWordFileLines][dimension];
        inWordFile.close();
        inWordFile = new BufferedReader(new FileReader(wordFile));

        BufferedReader inVectorFile = new BufferedReader(new FileReader(vectorFile));
      
        int i = -1;
        for(wordFileLine = inWordFile.readLine(); wordFileLine != null; wordFileLine = inWordFile.readLine()){
            i++;
            vectorFileLine = inVectorFile.readLine();
            WordToNum.put(wordFileLine, i);
            NumToWord.put(i, wordFileLine);
            String[] readWordVector = vectorFileLine.split("\\s+");
            for(int j = 0; j < dimension; j++)
                wordVector[i][j]=Double.parseDouble(readWordVector[j]);
        }
    }
}