package kr.kaist.sentence.embedding.execution;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import kr.kaist.sentence.embedding.structure.Batch;
import kr.kaist.sentence.embedding.structure.Document;
import kr.kaist.sentence.embedding.structure.MultiThread;
import kr.kaist.sentence.embedding.structure.Node;
import kr.kaist.sentence.embedding.structure.Tree;
import kr.kaist.sentence.embedding.util.ReadData;
import kr.kaist.sentence.embedding.util.TreeFactory;

public class ModelTrainer {
	
    static double Q;
    static double alpha;
    static int dimension;
    static int numThread;
    
    public static Vector<Tree>allTree = new Vector<Tree>();
    public static Vector<Document>allDocument = new Vector<Document>();
    public static Vector<Batch>allBatch = new Vector<Batch>();
    
    public static double[][] wordVector;
    public static double[][] weightMatrix;
    public static double[] bias;
    
    public static double[][] deltaWeightMatrix;
    public static double[] deltaBias;
    
    public static double[][] adaWeightMatrix;
    public static double[] adaBias;
    
    static double[][][] biasTree;
    static double[][][][] weightMatrixTree;
    static double[][][][] wordVectorTree;
    
	public static void main(String[] args) throws Exception {
        String vocabularyFile = "";
        String wordVectorFile = "";
        String parameterFile = "";
        String resultFile = "";
        String inputDirectory = "";
        boolean parameterFileFlag = false;
        for(int i=0;i<args.length;i++){
            if(args[i].equals("-dimension"))
                dimension = Integer.parseInt(args[i+1]);
            if(args[i].equals("-thread"))
                numThread = Integer.parseInt(args[i+1]);
                // thread_num for java parallel
            if(args[i].equals("-term_file"))
                vocabularyFile = args[i+1];
            if(args[i].equals("-vect_file"))
                wordVectorFile = args[i+1];
            if(args[i].equals("-param_file")){
                parameterFile=args[i+1];
                parameterFileFlag = true;
            }
            if(args[i].equals("-Q"))
                Q = Double.parseDouble(args[i+1]);
                // regularizer 
            if(args[i].equals("-alpha"))
                alpha = Double.parseDouble(args[i+1]);
                // learning rate
            if(args[i].equals("-input_dir"))
                inputDirectory = args[i+1];
            if(args[i].equals("-resultFile"))
                resultFile = args[i+1];
        }
        ReadData data = new ReadData(dimension);
        data.readWordVector(vocabularyFile, wordVectorFile);
        data.read(inputDirectory);
        data.getBatch();
        allTree = data.allTree;
        allDocument = data.allDocument;
        allBatch = data.allBatch;
        wordVector = data.wordVector;
        init();
        
        int iterationIndex = 0;
        while(true) {
        	iterationIndex++;
            System.out.println("iteration " + iterationIndex);
            System.out.println("predicted batch size: " + allBatch.size());
            for(int i = 0; i < allBatch.size(); i++) {
                System.out.println("now, " + i + "th batch has been processed. Total Batch number: " + allBatch.size());
                setDelta(); //set derivative to 0
                gradientDecent(i); //gradient decent
                update(); //parameter update
            }
            String fileName=resultFile 
            		+ "_iter" + Integer.toString(iterationIndex)
            		+ "_dimension" + Integer.toString(dimension)
            		+ "_Q" + Double.toString(Q)
            		+ ".txt";
            save(fileName); //save results
        }
	}
	
    public static void init() {
        weightMatrix = new double[dimension][dimension * 2];
        bias = new double[dimension];
        
        deltaWeightMatrix = new double[dimension][dimension * 2];
        deltaBias = new double[dimension];
        
        adaWeightMatrix = new double[dimension][dimension * 2];
        adaBias = new double[dimension];
        
        double epso=Math.sqrt(6)/(Math.sqrt(dimension));
        Random r = new Random();
        
        for(int i = 0; i < dimension; i++) {
            bias[i] = r.nextDouble() * 2 * epso - epso;
            for(int j = 0; j < dimension * 2; j++){
                weightMatrix[i][j] = r.nextDouble() * 2 * epso - epso;
            }
        }
    }
    
    public static void setDelta(){
        // set derivative to 0
        for(int i = 0; i < dimension; i++) {
            deltaBias[i] = 0;
            for(int j = 0; j < dimension * 2; j++){
                deltaWeightMatrix[i][j] = 0;
            }
        }
    }
    
    public static void gradientDecent(int batchIndex)throws Exception{
        Batch batch=allBatch.get(batchIndex);
        updateBatch(batchIndex);
        weightMatrixTree = getWeightMatrixTree(batch);	//	derivative of tree vector with respective to word_W
        biasTree = getBiasTree(batch);	//	derivative of tree vector with respective to word_b

        ExecutorService executor = Executors.newFixedThreadPool(numThread);	// multi thread
        List<Future<MultiThread>>list = new ArrayList<Future<MultiThread>>();
        //GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();

        for(int treeIndex : batch.treeList) {
            Tree tree = allTree.get(treeIndex);
            Callable<MultiThread>worker=new CallableHi(tree, batch);
            Future<MultiThread>submit=executor.submit(worker);
            list.add(submit);
        }
        for (Future<MultiThread> future : list) {
            MultiThread thread = future.get();
            for(int i = 0; i < dimension; i++) {
                deltaBias[i] += thread.derivativeBias[i];
                for(int j = 0; j < 2 * dimension; j++)
                    deltaWeightMatrix[i][j] += thread.derivativeWeightMatrix[i][j];
            }
        }
        executor.shutdown();

        double l = (double)1 / batch.treeList.size();
        double T = 2 * Q / allTree.size();

        deltaBias = getVectorScalar(l, deltaBias);
        deltaWeightMatrix = getWeightMatrixScalar(l, deltaWeightMatrix);
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < 2 * dimension; j++)
                deltaWeightMatrix[i][j] += T * weightMatrix[i][j];	// regulizer
    }

    public static void updateBatch(int batchIndex) {
        //update all node vectors within trees in the batch
        Batch batch = allBatch.get(batchIndex);
        for(int treeIndex : batch.treeList) {
            TreeFactory treeFactory = new TreeFactory();
        	Tree tree = allTree.get(treeIndex);
            treeFactory.getVector(0, weightMatrix, bias, wordVector, tree);
            allTree.set(treeIndex, tree);
        }
    }
    
    public static double[][][][] getWeightMatrixTree(Batch batch) throws Exception {
        double[][][][] allTreeWeightMatrix;
        allTreeWeightMatrix = new double[batch.treeList.size()][dimension][dimension * 2][dimension];
        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        List<Future<double[][][]>>list = new ArrayList<Future<double[][][]>>();
        int currentTreeIndex = 0;
        for(int i = 0; i < batch.treeList.size(); i++) {
            int treeIndex = batch.treeList.get(i);
            if(treeIndex >= 0) {
                Tree tree = allTree.get(treeIndex);
                Callable<double[][][]> worker = new CallableWeightMatrix(tree);
                Future<double[][][]> submit = executor.submit(worker);
                list.add(submit);
            }
        }
        for (Future<double[][][]> future : list) {
            for(int i=0;i<dimension;i++)
                for(int j=0;j<2*dimension;j++)
                	allTreeWeightMatrix[currentTreeIndex][i][j]=future.get()[i][j];
            currentTreeIndex++;
        }
        executor.shutdown();
        
        return allTreeWeightMatrix;
    }
    
    public static double[][][] getBiasTree(Batch batch) throws Exception {
        double[][][] allTreeBias = new double[batch.treeList.size()][dimension][dimension];
        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        List<Future<double[][]>>list = new ArrayList<Future<double[][]>>();
        int currentTreeIndex = 0;
        for(int i = 0; i < batch.treeList.size(); i++) {
            int treeIndex = batch.treeList.get(i);
            if(treeIndex >= 0) {
                Tree tree = allTree.get(treeIndex);
                Callable<double[][]> worker = new CallableBias(tree);
                Future<double[][]> submit = executor.submit(worker);
                list.add(submit);
            }
        }
        for (Future<double[][]> future : list) {
            for(int i = 0; i < dimension; i++)
                allTreeBias[currentTreeIndex][i] = future.get()[i];
            currentTreeIndex++;
        }
        executor.shutdown();
        
        return allTreeBias;
    }
    
    public static class CallableWeightMatrix implements Callable<double[][][]>{
        Tree tree;
        public CallableWeightMatrix(Tree oldTree) {
            this.tree = oldTree;
        }
        public double[][][] call() throws Exception {
            double[][][] result = new double[dimension][dimension * 2][dimension];
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            for(int i = 0; i < dimension; i++)
                for(int j = 0; j < 2 * dimension; j++) {
                    result[i][j] = gradientDecentCalculator.gradientDecentWeightMatrix(tree, 0, i, j);
                }
            return result;
        }
    }
    public static class CallableBias implements Callable<double[][]>{
        Tree tree;
        public CallableBias(Tree oldTree) {
            this.tree=oldTree;
        }
        public double[][] call()throws Exception {
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            double[][] result = new double[dimension][dimension];
            for(int i = 0; i < dimension; i++)
                result[i] = gradientDecentCalculator.gradientDecentBias(tree,0,i);
            return result;
        }
    }
    
    public static class CallableHi implements Callable<MultiThread>{
        Batch batch;
        Tree tree;
        public CallableHi(Tree tree, Batch batch){
            this.batch=batch;
        	this.tree = tree;
        }
        public MultiThread call()throws Exception {
            MultiThread thread=new MultiThread(dimension);
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            
            double[][] T = new double[dimension][dimension];
            for(int i = 0; i < dimension; i++)
                for(int j = 0; j < dimension; j++) {
                    T[i][j] = weightMatrix[i][j] * bias[i];	// 이 부분이 문제
                }
            thread.derivativeBias = gradientDecentCalculator.getGradientDecentDerivativeBias(tree, batch, T);
            thread.derivativeWeightMatrix = gradientDecentCalculator.getGradientDecentDerivativeWeightMatrix(tree, batch, T);
            return thread;
        }
    }
    
    public static class GradientDecentCalculator {
        //get derivative
        public static double[] gradientDecentBias(Tree tree, int nodeIndex, int biasIndex) {
            //get derivative of tree (sentence) vector with respect to word_B
            Node node = tree.allNodes.get(nodeIndex);
            node.calculatedVector = new double[dimension];
            if(node.isLeaf){
                for(int i = 0; i < dimension; i++)
                	node.calculatedVector[i] = 0;
                return node.calculatedVector;
            } else if(node.childrenList.size() == 1) {
                int child = node.childrenList.get(0);
                node.calculatedVector = gradientDecentBias(tree, child, biasIndex);
                return node.calculatedVector;
            } else {
                int leftChildIndex = node.childrenList.get(0);
                int rightChildIndex = node.childrenList.get(1);
                double[] leftWordVector = new double[dimension];
                double[] rightWordVector = new double[dimension];
                leftWordVector = gradientDecentBias(tree, leftChildIndex, biasIndex);
                rightWordVector = gradientDecentBias(tree, rightChildIndex, biasIndex);
                for(int t = 0; t < dimension; t++){
                    for(int i = 0; i < dimension; i++) {
                        node.calculatedVector[t] += weightMatrix[t][i] * leftWordVector[i];
                        node.calculatedVector[t] += weightMatrix[t][dimension+i] * rightWordVector[i];
                    }
                }
                node.calculatedVector[biasIndex]++;
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] *= node.gradientVector[i];
                return node.calculatedVector;
            }
        }
        
        public static double[] gradientDecentWeightMatrix(Tree tree, int nodeIndex, int index1, int index2){
            //get derivative of tree(sentence) vector with respect to word_W
            Node node = tree.allNodes.get(nodeIndex);
            node.calculatedVector = new double[dimension];
            if(node.isLeaf) {
                for(int i = 0; i < dimension; i++)
                	node.calculatedVector[i] = 0;
                return node.calculatedVector;
            } else if(node.childrenList.size() == 1) {
                int childIndex = node.childrenList.get(0);
                node.calculatedVector = gradientDecentWeightMatrix(tree, childIndex, index1, index2);
                return node.calculatedVector;
            } else {
                int leftChildIndex = node.childrenList.get(0);
                int rightChildIndex = node.childrenList.get(1);
                double[] leftWordVector = new double[dimension];
                double[] rightWordVector = new double[dimension];
                leftWordVector = gradientDecentWeightMatrix(tree, leftChildIndex, index1, index2);
                rightWordVector = gradientDecentWeightMatrix(tree, rightChildIndex, index1, index2);
                for(int t = 0; t < dimension; t++) {
                    for(int i = 0; i < dimension; i++)
                        node.calculatedVector[t] += weightMatrix[t][i] * leftWordVector[i];
                    for(int i = 0; i < dimension; i++)
                        node.calculatedVector[t] += weightMatrix[t][dimension+i] * rightWordVector[i];
                }
                if(index2 < dimension)
                	node.calculatedVector[index1] += tree.allNodes.get(leftChildIndex).vector[index2];
                else 
                	node.calculatedVector[index1] += tree.allNodes.get(rightChildIndex).vector[index2 - dimension];
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] *= node.gradientVector[i];
                return node.calculatedVector;
            }
        }
        
        public static double[] getGradientDecentDerivativeBias(Tree tree, Batch batch, double[][] T) {
            //get derivative of tree value with respect to word_b
            double[] resultBias = new double[dimension];
            double[][] join = new double[dimension][dimension];
            
            int treeIndex = batch.treeList.indexOf(allTree.indexOf(tree));
            for(int i = 0; i < dimension; i++) {
            	for(int j = 0; j < dimension; j++) {
            		join[i][j] = biasTree[treeIndex][i][j];
            	}
            }
            resultBias = sumMatrixLine(multiplyMatrixMatrix(join, T));

            return resultBias;
        }

        static public double[][] getGradientDecentDerivativeWeightMatrix (Tree tree, Batch batch, double[][] T){
            //get derivative of clique value with respect to word_W
            double[][] resultWeightMatrix = new double[dimension][2 * dimension];
            double[][][] join = new double[dimension][2 * dimension][dimension];
            int treeIndex = batch.treeList.indexOf(allTree.indexOf(tree));
            for(int i = 0; i < dimension; i++) {
            	for(int j = 0; j < 2 * dimension; j++) {
            		for(int k = 0; k < dimension; k++)
            			join[i][j][k] = weightMatrixTree[treeIndex][i][j][k];
            	}
            }
            for(int i = 0; i < dimension; i++)
            	resultWeightMatrix[i] = sumMatrixLine(multiplyMatrixMatrix(join[i], T));
            return resultWeightMatrix;
        }
    }
    
    public static void update(){//update parameters

        for(int i = 0; i < dimension; i++) {
            adaBias[i] += Math.pow(deltaBias[i], 2);
            bias[i] -= alpha * deltaBias[i] / Math.sqrt(adaBias[i]);
            for(int j = 0; j < 2 * dimension; j++) {
                adaWeightMatrix[i][j] += Math.pow(deltaWeightMatrix[i][j], 2);
                weightMatrix[i][j] -= alpha * deltaWeightMatrix[i][j] / Math.sqrt(adaWeightMatrix[i][j]);
            }
        }
    }

    public static void save(String fileName)throws IOException{
        //save parameters
        FileWriter fw = new FileWriter(fileName);

        fw.write("weightMatrix\n");
        for(int i = 0; i < dimension; i++){
            for(int j = 0; j < 2 * dimension; j++)
                fw.write(weightMatrix[i][j] + " ");
            fw.write("\n");
        }

        fw.write("bias\n");
        for(int i = 0; i < dimension; i++)
            fw.write(bias[i] + " ");
        fw.write("\n");

        fw.close();
    }

    public static double[] getVectorScalar(double a1, double[] a2){
        //multiple vector a2 by scalar a1
        double[] c = new double[a2.length];
        for(int i = 0; i < a2.length; i++)
            c[i] = a1 * a2[i];
        return c;
    }
    
    public static double[][] getWeightMatrixScalar(double a1, double[][] a2){
        //multiple matrix a2 by scalar a1
        double[][] c = new double[a2.length][a2[0].length];
        for(int i = 0; i < a2.length; i++)
            for(int j = 0; j < a2[0].length; j++)
                c[i][j] = a1 * a2[i][j];
        return c;
    }
    
    public static double[][] multiplyMatrixMatrix(double[][] A, double[][] B) {
        if(A[0].length != B[0].length)
        	System.out.println("Matrix Dimensions not Constant");
        double[][]C = new double[A.length][B.length];
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < B.length; j++) {
                for(int k = 0; k < A[0].length; k++) {
                    C[i][j] += A[i][k] * B[j][k];
                }
            }
        }
        return C;
    }
    
    public static double[] sumMatrixLine(double[][] a) {
        double[] L = new double[a.length];
        for(int i = 0; i < a.length; i++)
            for(int j = 0; j < a[0].length; j++)
                L[i] += a[i][j];
        return L;
    }
}
