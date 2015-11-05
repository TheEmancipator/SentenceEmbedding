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
import kr.kaist.sentence.embedding.util.RSTTreeFactory;
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

    public static double[][] documentWeightMatrix;
    public static double[] documentBias;

    public static double[] outputU;
    public static double outputBias;
    
    
    public static double[][] deltaWordVector;
    public static double[][] deltaWeightMatrix;
    public static double[] deltaBias;

    public static double[][] deltaDocumentWeightMatrix;
    public static double[] deltaDocumentBias;

    public static double[] deltaOutputU;
    public static double deltaOutputBias;
    
    public static double[][] adaWordVector;
    public static double[][] adaWeightMatrix;
    public static double[] adaBias;
    
    public static double[][] adaDocumentWeightMatrix;
    public static double[] adaDocumentBias;

    public static double[] adaOutputU;
    public static double adaOutputBias;
    
    static double[][][] biasTree;
    static double[][][][] weightMatrixTree;
    //static double[][][] documentBiasTree;
    //static double[][][][] documentWeightMatrixTree;
    
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
            if(args[i].equals("-result_file"))
                resultFile = args[i+1];
        }
        ReadData data = new ReadData(dimension);
        data.readWordVector(vocabularyFile, wordVectorFile);
        data.read(inputDirectory);
        data.getBatch(25);
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
                update(); //parameter update    // 이거 해야 함//////////////////////
                ///////////////////////////////////////////
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
        
        documentWeightMatrix = new double[dimension][dimension * 2];
        documentBias = new double[dimension];
        
        outputU = new double[dimension];
        
        deltaWeightMatrix = new double[dimension][dimension * 2];
        deltaBias = new double[dimension];
        
        deltaDocumentWeightMatrix = new double[dimension][dimension * 2];
        deltaDocumentBias = new double[dimension];
        
        deltaOutputU = new double[dimension];
        
        adaWeightMatrix = new double[dimension][dimension * 2];
        adaBias = new double[dimension];

        adaDocumentWeightMatrix = new double[dimension][dimension * 2];
        adaDocumentBias = new double[dimension];
        
        adaOutputU = new double[dimension];
        
        double epso=Math.sqrt(6)/(Math.sqrt(dimension));
        Random r = new Random();
        
        for(int i = 0; i < dimension; i++) {
            outputU[i] = r.nextDouble() *2 * epso - epso;
            //bias[i] = r.nextDouble() * 2 * epso - epso;
            for(int j = 0; j < dimension * 2; j++){
                weightMatrix[i][j] = r.nextDouble() * 2 * epso - epso;
                documentWeightMatrix[i][j] = r.nextDouble() * 2 * epso - epso;
            }
        }
    }
    
    public static void setDelta(){
        // set derivative to 0
        deltaOutputBias = 0;
        for(int i = 0; i < dimension; i++) {
            deltaDocumentBias[i] = 0;
            deltaOutputU[i] = 0;
            for(int j = 0; j < dimension * 2; j++){
                deltaWeightMatrix[i][j] = 0;
                deltaDocumentWeightMatrix[i][j] = 0;
            }
        }
    }
    
    public static void gradientDecent(int batchIndex)throws Exception{
        updateBatch(batchIndex);
        Batch batch=allBatch.get(batchIndex);
        weightMatrixTree = getWeightMatrixTree(batch);  //  derivative of tree vector with respective to word_W
        biasTree = getBiasTree(batch);  //  derivative of tree vector with respective to word_b
        //documentWeightMatrixTree = getDocumentWeightMatrixTree(batch); //   derivative of tree vector with respective to document_W
        //documentBiasTree = getDocumentBiasTree(batch); //   derivative of tree vector with respective to document_b

        ExecutorService executor = Executors.newFixedThreadPool(numThread); // multi thread
        List<Future<MultiThread>>list = new ArrayList<Future<MultiThread>>();
        //GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
        
        for(int documentIndex : batch.documentList) {
            System.out.println(documentIndex + " document in batch has been processed.");
            Document document = allDocument.get(documentIndex);
            deltaOutputBias += (document.inferredClass - document.tag);
            //System.out.println("deltaOutputBias: " + deltaOutputBias);
            //System.out.println("inferredClass: " + document.inferredClass);
            //System.out.println("tag: " + document.tag);
            Callable<MultiThread>worker=new CallableHi(document, batch);
            Future<MultiThread>submit=executor.submit(worker);
            list.add(submit);
        }
        for (Future<MultiThread> future : list) {
            MultiThread thread = future.get();
            for(int i = 0; i < dimension; i++) {
                deltaOutputU[i] += thread.derivativeOutputU[i];
                //System.out.println("deltaOutputU: " + deltaOutputU[i]);
                deltaDocumentBias[i] = thread.derivativeDocumentBias[i];
                //System.out.println("deltaDocumentBias: " + deltaDocumentBias[i]);
                deltaBias[i] += thread.derivativeBias[i];
                //System.out.println("deltaBias: " + deltaBias[i]);
                for(int j = 0; j < 2 * dimension; j++) {
                    deltaWeightMatrix[i][j] += thread.derivativeWeightMatrix[i][j];
                    deltaDocumentWeightMatrix[i][j] += thread.derivativeDocumentWeightMatrix[i][j];
                }
            }
        }
        executor.shutdown();
        deltaOutputBias /= batch.documentList.size();

        
        double l = (double)1 / batch.documentList.size();
        double T = (double)Q / (2 *allDocument.size());

        deltaOutputU = multiplyScalarVector(l, deltaOutputU);
        for(int i = 0; i < dimension; i ++) // regularizer
            deltaOutputU[i] += T * outputU[i];
        
        deltaDocumentBias = multiplyScalarVector(l, deltaDocumentBias);
        
        deltaDocumentWeightMatrix = multiplyScalarMatrix(l, deltaDocumentWeightMatrix);
        for(int i = 0; i < dimension; i ++) // regularizer
            for(int j = 0; j < 2 * dimension; j++)
                deltaDocumentWeightMatrix[i][j] += T * documentWeightMatrix[i][j];
        
        deltaBias = multiplyScalarVector(l, deltaBias);
        deltaWeightMatrix = multiplyScalarMatrix(l, deltaWeightMatrix);
        for(int i = 0; i < dimension; i++)
            for(int j = 0; j < 2 * dimension; j++)
                deltaWeightMatrix[i][j] += T * weightMatrix[i][j];  // regulizer
    }

    public static void updateBatch(int batchIndex) {
        //update all node vectors within trees in the batch
        Batch batch = allBatch.get(batchIndex);
        for(int treeIndex : batch.treeList) {
            TreeFactory treeFactory = new TreeFactory();
            Tree tree = allTree.get(treeIndex);
            treeFactory.getTreeVector(0, weightMatrix, bias, wordVector, tree);
            allTree.set(treeIndex, tree);
        }
        for(int documentIndex : batch.documentList) {
            Document document = allDocument.get(documentIndex);
            RSTTreeFactory rstTreeFactory = new RSTTreeFactory();
            rstTreeFactory.getRSTTreeVector(0, documentWeightMatrix, documentBias, allTree, document);
            document.inferredClass = sigmod(dot(tanhVector(document.allNodes.get(0).vector), outputU) + outputBias);
            allDocument.set(documentIndex, document);
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
            for(int i = 0; i < dimension; i++)
                for(int j = 0; j < 2 * dimension; j++)
                    allTreeWeightMatrix[currentTreeIndex][i][j] = future.get()[i][j];
            currentTreeIndex++;
            //System.out.println(currentTreeIndex + "'s tree in batch has been processed. (WM)");
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
            //System.out.println(currentTreeIndex + "'s tree in batch has been processed. (b)");
        }
        executor.shutdown();
        
        return allTreeBias;
    }
    
/*    public static double[][][][] getDocumentWeightMatrixTree(Batch batch) throws Exception {
        double[][][][] allDocumentTreeWeightMatrix;
        allDocumentTreeWeightMatrix = new double[batch.documentList.size()][dimension][dimension * 2][dimension];
        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        List<Future<double[][][]>>list = new ArrayList<Future<double[][][]>>();
        int currentDocumentIndex = 0;
        for(int i = 0; i < batch.documentList.size(); i++) {
            int documentIndex = batch.documentList.get(i);
            if(documentIndex >= 0) {
                Document document = allDocument.get(documentIndex);
                Callable<double[][][]> worker = new CallableDocumentWeightMatrix(document);
                Future<double[][][]> submit = executor.submit(worker);
                list.add(submit);
            }
        }
        for (Future<double[][][]> future : list) {
            for(int i = 0; i < dimension; i++)
                for(int j = 0; j < 2 * dimension; j++)
                    allDocumentTreeWeightMatrix[currentDocumentIndex][i][j] = future.get()[i][j];
            currentDocumentIndex++;
            //System.out.println(currentDocumentIndex + "'s tree in batch has been processed. (DWM)");
        }
        executor.shutdown();
        
        return allDocumentTreeWeightMatrix;
    }
    
    public static double[][][] getDocumentBiasTree(Batch batch) throws Exception {
        double[][][] allDocumentTreeBias = new double[batch.documentList.size()][dimension][dimension];
        ExecutorService executor = Executors.newFixedThreadPool(numThread);
        List<Future<double[][]>>list = new ArrayList<Future<double[][]>>();
        int currentDocumentIndex = 0;
        for(int i = 0; i < batch.documentList.size(); i++) {
            int documentIndex = batch.documentList.get(i);
            if(documentIndex >= 0) {
                Document document = allDocument.get(documentIndex);
                Callable<double[][]> worker = new CallableDocumentBias(document);
                Future<double[][]> submit = executor.submit(worker);
                list.add(submit);
            }
        }
        for (Future<double[][]> future : list) {
            for(int i = 0; i < dimension; i++)
                allDocumentTreeBias[currentDocumentIndex][i] = future.get()[i];
            currentDocumentIndex++;
            //System.out.println(currentDocumentIndex + "'s tree in batch has been processed. (Db)");
        }
        executor.shutdown();
        
        return allDocumentTreeBias;
    }*/
    
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
                    result[i][j] = gradientDecentCalculator.getGradientDecentWeightMatrix(tree, 0, i, j);
                }
            return result;
        }
    }
    public static class CallableBias implements Callable<double[][]>{
        Tree tree;
        public CallableBias(Tree oldTree) {
            this.tree = oldTree;
        }
        public double[][] call()throws Exception {
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            double[][] result = new double[dimension][dimension];
            for(int i = 0; i < dimension; i++)
                result[i] = gradientDecentCalculator.getGradientDecentBias(tree,0,i);
            return result;
        }
    }
    
    public static class CallableDocumentWeightMatrix implements Callable<double[][][]>{
        Document document;
        public CallableDocumentWeightMatrix(Document oldDocument) {
            this.document = oldDocument;
        }
        public double[][][] call() throws Exception {
            double[][][] result = new double[dimension][dimension * 2][dimension];
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            for(int i = 0; i < dimension; i++)
                for(int j = 0; j < 2 * dimension; j++) {
                    result[i][j] = gradientDecentCalculator.getGradientDecentDocumentWeightMatrix(document, 0, i, j);
                }
            return result;
        }
    }
    
    public static class CallableDocumentBias implements Callable<double[][]>{
        Document document;
        public CallableDocumentBias(Document oldDocument) {
            this.document = oldDocument;
        }
        public double[][] call()throws Exception {
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            double[][] result = new double[dimension][dimension];
            for(int i = 0; i < dimension; i++)
                result[i] = gradientDecentCalculator.getGradientDecentDocumentBias(document, 0, i);
            return result;
        }
    }
    
    public static class CallableHi implements Callable<MultiThread>{
        Batch batch;
        Document document;
        public CallableHi(Document document, Batch batch){
            this.batch=batch;
            this.document = document;
        }
        public MultiThread call()throws Exception {
            MultiThread thread=new MultiThread(dimension);
            GradientDecentCalculator gradientDecentCalculator = new GradientDecentCalculator();
            thread.derivativeOutputU = gradientDecentCalculator.getGradientDecentDerivativeOutputU(document);

            thread.derivativeDocumentBias = gradientDecentCalculator.getGradientDecentDerivativeDocumentBias(document);
            /*for(int i = 0; i < dimension; i++)
                System.out.print("tdDoucument Bias: " + thread.derivativeDocumentBias[i] + " ");
            System.out.print("\n");*/
            thread.derivativeDocumentWeightMatrix = gradientDecentCalculator.getGradientDecentDerivativeDocumentWeightMatrix(document, thread.derivativeDocumentBias);
            
            double[][] T = new double[dimension][2 * dimension];
            for(int i = 0; i < dimension; i++)
                for(int j = 0; j < 2 * dimension; j++)
                    T[i][j] = documentWeightMatrix[i][j] * thread.derivativeDocumentBias[i];
            thread.derivativeBias = gradientDecentCalculator.getGradientDecentDerivativeBias(document, batch, T);
            /*for(int i = 0; i < dimension; i++)
                System.out.print("tdBias: " + thread.derivativeBias[i] + " ");
            System.out.print("\n");*/
            thread.derivativeWeightMatrix = gradientDecentCalculator.getGradientDecentDerivativeWeightMatrix(document, batch, T);
            return thread;
        }
    }
    
    public static class GradientDecentCalculator {
        //get derivative
        
        public static double[] getGradientDecentBias(Tree tree, int nodeIndex, int biasIndex) {
            //get derivative of tree (sentence) vector with respect to word_B
            Node node = tree.allNodes.get(nodeIndex);
            node.calculatedVector = new double[dimension];
            if(node.isLeaf){
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] = 0;
                return node.calculatedVector;
            } else if(node.childrenList.size() == 1) {
                int child = node.childrenList.get(0);
                node.calculatedVector = getGradientDecentBias(tree, child, biasIndex);
                return node.calculatedVector;
            } else {
                int leftChildIndex = node.childrenList.get(0);
                int rightChildIndex = node.childrenList.get(1);
                double[] leftWordVector = new double[dimension];
                double[] rightWordVector = new double[dimension];
                leftWordVector = getGradientDecentBias(tree, leftChildIndex, biasIndex);
                rightWordVector = getGradientDecentBias(tree, rightChildIndex, biasIndex);
                for(int t = 0; t < dimension; t++){
                    for(int i = 0; i < dimension; i++) {
                        node.calculatedVector[t] += weightMatrix[t][i] * leftWordVector[i];
                        node.calculatedVector[t] += weightMatrix[t][dimension+i] * rightWordVector[i];
                    }
                }
                node.calculatedVector[biasIndex]++;
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] *= node.vectorDerivative[i];
                return node.calculatedVector;
            }
        }
        
        public static double[] getGradientDecentWeightMatrix(Tree tree, int nodeIndex, int index1, int index2){
            //get derivative of tree(sentence) vector with respect to word_W
            Node node = tree.allNodes.get(nodeIndex);
            node.calculatedVector = new double[dimension];
            if(node.isLeaf) {
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] = 0;
                return node.calculatedVector;
            } else if(node.childrenList.size() == 1) {
                int childIndex = node.childrenList.get(0);
                node.calculatedVector = getGradientDecentWeightMatrix(tree, childIndex, index1, index2);
                return node.calculatedVector;
            } else {
                int leftChildIndex = node.childrenList.get(0);
                int rightChildIndex = node.childrenList.get(1);
                double[] leftWordVector = new double[dimension];
                double[] rightWordVector = new double[dimension];
                leftWordVector = getGradientDecentWeightMatrix(tree, leftChildIndex, index1, index2);
                rightWordVector = getGradientDecentWeightMatrix(tree, rightChildIndex, index1, index2);
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
                    node.calculatedVector[i] *= node.vectorDerivative[i];
                return node.calculatedVector;
            }
        }
        
        public static double[] getGradientDecentDocumentBias(Document document, int nodeIndex, int biasIndex) {
            //get derivative of tree (sentence) vector with respect to Document_B
            Node node = document.allNodes.get(nodeIndex);
            node.calculatedVector = new double[dimension];
            if(node.isLeaf){
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] = 0;
                return node.calculatedVector;
            } else if(node.childrenList.size() == 1) {
                int child = node.childrenList.get(0);
                node.calculatedVector = getGradientDecentDocumentBias(document, child, biasIndex);
                return node.calculatedVector;
            } else {
                int leftChildIndex = node.childrenList.get(0);
                int rightChildIndex = node.childrenList.get(1);
                double[] leftWordVector = new double[dimension];
                double[] rightWordVector = new double[dimension];
                leftWordVector = getGradientDecentDocumentBias(document, leftChildIndex, biasIndex);
                rightWordVector = getGradientDecentDocumentBias(document, rightChildIndex, biasIndex);
                for(int t = 0; t < dimension; t++){
                    for(int i = 0; i < dimension; i++) {
                        node.calculatedVector[t] += weightMatrix[t][i] * leftWordVector[i];
                        node.calculatedVector[t] += weightMatrix[t][dimension+i] * rightWordVector[i];
                    }
                }
                node.calculatedVector[biasIndex]++;
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] *= node.vectorDerivative[i];
                return node.calculatedVector;
            }
        }
        
        public static double[] getGradientDecentDocumentWeightMatrix(Document document, int nodeIndex, int index1, int index2){
            //get derivative of tree(sentence) vector with respect to document_W
            Node node = document.allNodes.get(nodeIndex);
            node.calculatedVector = new double[dimension];
            if(node.isLeaf) {
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] = 0;
                return node.calculatedVector;
            } else if(node.childrenList.size() == 1) {
                int childIndex = node.childrenList.get(0);
                node.calculatedVector = getGradientDecentDocumentWeightMatrix(document, childIndex, index1, index2);
                return node.calculatedVector;
            } else {
                int leftChildIndex = node.childrenList.get(0);
                int rightChildIndex = node.childrenList.get(1);
                double[] leftWordVector = new double[dimension];
                double[] rightWordVector = new double[dimension];
                leftWordVector = getGradientDecentDocumentWeightMatrix(document, leftChildIndex, index1, index2);
                rightWordVector = getGradientDecentDocumentWeightMatrix(document, rightChildIndex, index1, index2);
                for(int t = 0; t < dimension; t++) {
                    for(int i = 0; i < dimension; i++)
                        node.calculatedVector[t] += weightMatrix[t][i] * leftWordVector[i];
                    for(int i = 0; i < dimension; i++)
                        node.calculatedVector[t] += weightMatrix[t][dimension+i] * rightWordVector[i];
                }
                if(index2 < dimension)
                    node.calculatedVector[index1] += document.allNodes.get(leftChildIndex).vector[index2];
                else 
                    node.calculatedVector[index1] += document.allNodes.get(rightChildIndex).vector[index2 - dimension];
                for(int i = 0; i < dimension; i++)
                    node.calculatedVector[i] *= node.vectorDerivative[i];
                return node.calculatedVector;
            }
        }
        
        public static double[] getGradientDecentDerivativeOutputU(Document document) {
            //get derivative of clique value with respect to sen_U
            double[]for_U = new double[dimension];
            double t = document.inferredClass - document.tag;
            for_U = multiplyScalarVector(t, tanhVector(document.allNodes.get(0).vector));
            return for_U;
        }
        
        public static double[] getGradientDecentDerivativeDocumentBias(Document document) {
            double[] resultBias = new double[dimension];
            
            double t = document.inferredClass - document.tag;
            resultBias = multiplyScalarVector(t, dotDot(outputU, derivativeTanhVector(tanhVector(document.allNodes.get(0).vector))));

            return resultBias;
        }
        
        static public double[][] getGradientDecentDerivativeDocumentWeightMatrix(Document document, double[] derivativeDocumentBias){
            //get derivative of clique value with respect to sen_W
            double[][] resultWeightMatrix = new double[dimension][2 * dimension];
            
            Node node = document.allNodes.get(0);
            
            double[] firstChildVector = new double[dimension];
            double[] secondChildVector = new double[dimension];
            int firstChildIndex = node.childrenList.get(0);
            int secondChildIndex = node.childrenList.get(1);
            firstChildVector = document.allNodes.get(firstChildIndex).vector;
            secondChildVector = document.allNodes.get(secondChildIndex).vector;
            double[] concatenatedVector = new double[2 * dimension];
            // concatenate firstChildVector and secondChildVector
            for(int i = 0; i < dimension; i++)
                concatenatedVector[i] = firstChildVector[i];
            for(int i = 0; i < dimension; i++)
                concatenatedVector[dimension + i] = secondChildVector[i];
            
            resultWeightMatrix = vectorVectorMatrix(derivativeDocumentBias, concatenatedVector);

            return resultWeightMatrix;
        }
        
        public static double[] getGradientDecentDerivativeBias(Document document, Batch batch, double[][] T) {
            //get derivative of tree value with respect to word_b
            double[] resultBias = new double[dimension];
            double[][] join = new double[dimension][dimension];
            
            for(int i = 0; i < document.treeList.size(); i++) {
                int treeIndex = document.treeList.get(i);
                if(treeIndex < 0)
                    continue;
                int index = batch.treeList.indexOf(treeIndex);
                for(int j = 0; j < dimension; j++)
                    for(int k = 0; k < dimension; k++)
                        join[j][k] = biasTree[index][j][k]; 
            }

            resultBias = sumMatrixLine(multiplyMatrixMatrix(join, T));

            return resultBias;
        }

        public static double[][] getGradientDecentDerivativeWeightMatrix (Document document, Batch batch, double[][] T){
            //get derivative of clique value with respect to word_W
            double[][] resultWeightMatrix = new double[dimension][2 * dimension];
            double[][][] join = new double[dimension][2 * dimension][dimension];
            for(int i = 0; i < document.treeList.size(); i++) {
                int treeIndex = document.treeList.get(i);
                if(treeIndex < 0)
                    continue;
                int index = batch.treeList.indexOf(treeIndex);
                for(int j = 0; j < dimension; j++) 
                    for(int k = 0; k < 2 * dimension; k++)
                        for(int m = 0; m < dimension; m++)
                            join[j][k][m] = weightMatrixTree[index][j][k][m];
            }

            for(int i = 0; i < dimension; i++)
                resultWeightMatrix[i] = sumMatrixLine(multiplyMatrixMatrix(join[i], T));
            return resultWeightMatrix;
        }
    }
///////////////////////////////////////////////////////////////////////////////////
//finished    
    public static void update(){//update parameters
        
        for(int i = 0; i < dimension; i++) {
            adaOutputU[i] += Math.pow(deltaOutputU[i], 2);
            outputU[i] -= alpha * deltaOutputU[i] / Math.sqrt(adaOutputU[i]);
            
            adaDocumentBias[i] += Math.pow(deltaDocumentBias[i], 2);
            documentBias[i] -= alpha * deltaDocumentBias[i] / Math.sqrt(adaDocumentBias[i]);
            // 여기...............
            //System.out.println("deltaDocumentBias: " + deltaDocumentBias[i]);
            //System.out.println("adaDocumentBias: " + adaDocumentBias[i]);
            //System.out.println("documentBias: " + documentBias[i]);

            adaBias[i] += Math.pow(deltaBias[i], 2);
            bias[i] -= alpha * deltaBias[i] / Math.sqrt(adaBias[i]);
            //System.out.println("bias: " + bias[i]);            

            for(int j = 0; j < 2 * dimension; j++) {
                adaDocumentWeightMatrix[i][j] += Math.pow(deltaDocumentWeightMatrix[i][j], 2);
                documentWeightMatrix[i][j] -= alpha * deltaDocumentWeightMatrix[i][j] / Math.sqrt(adaDocumentWeightMatrix[i][j]);
                
                adaWeightMatrix[i][j] += Math.pow(deltaWeightMatrix[i][j], 2);
                weightMatrix[i][j] -= alpha * deltaWeightMatrix[i][j] / Math.sqrt(adaWeightMatrix[i][j]);
            }
        }
        adaOutputBias += Math.pow(deltaOutputBias, 2);
        outputBias -= alpha * deltaOutputBias / Math.sqrt(adaOutputBias);
        //System.out.println("outputBias: " + outputBias);
    }

    public static void save(String fileName)throws IOException{
        //save parameters
        FileWriter fw = new FileWriter(fileName);

        fw.write("outputU\n");
        for(int i = 0; i < dimension; i++)
            fw.write(outputU[i] + " ");
        fw.write("\n");
        
        fw.write("outputBias\n");
        fw.write(outputBias + "\n");
        
        fw.write("documentWeightMatrix\n");
        for(int i = 0; i < dimension; i++){
            for(int j = 0; j < 2 * dimension; j++)
                fw.write(documentWeightMatrix[i][j] + " ");
            fw.write("\n");
        }

        fw.write("documentBias\n");
        for(int i = 0; i < dimension; i++)
            fw.write(documentBias[i] + " ");
        fw.write("\n");

        
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

    public static double[] multiplyScalarVector(double a1, double[] a2){
        //multiple vector a2 by scalar a1
        double[] c = new double[a2.length];
        for(int i = 0; i < a2.length; i++)
            c[i] = a1 * a2[i];
        return c;
    }
    
    public static double[][] multiplyScalarMatrix(double a1, double[][] a2){
        //multiple matrix a2 by scalar a1
        double[][] c = new double[a2.length][a2[0].length];
        for(int i = 0; i < a2.length; i++)
            for(int j = 0; j < a2[0].length; j++)
                c[i][j] = a1 * a2[i][j];
        return c;
    }
    
    public static double[][] multiplyMatrixMatrix(double[][] A, double[][] B) {
        if(A[0].length != B.length)
            System.out.println("Matrix Dimensions not Constant");
        double[][]C = new double[A.length][B[0].length];
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < B[0].length; j++) {
                for(int k = 0; k < A[0].length; k++) {
                    C[i][j] += A[i][k] * B[k][j];
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
    
    public static double[] multiplyVectorVector(double[]a1,double[]a2){
        // .dot
        double[] c = new double[a1.length];
        for(int i = 0; i < a1.length; i++)
            c[i] = a1[i] * a2[i];
        return c;
    }
    
    public static double[][] vectorVectorMatrix(double[]a1,double[]a2){
        double[][] G = new double[a1.length][a2.length];
        for(int i = 0; i < a1.length; i++)
            for(int j = 0; j < a2.length; j++)
                G[i][j] = a1[i] * a2[j];
        return G;
    }

    public static double dot(double[]a1,double[]a2){
        //dot-multiply
        double total=0;
        for(int i=0;i<a1.length;i++)
            total+=a1[i]*a2[i];
        return total;
    }

    public static double sigmod(double a){
        //sigmod function
        return 1/(1+Math.exp(-a));
    }

    public static double[] tanhVector (double []a1) {
        double[] A = new double[a1.length];
        for(int i = 0; i < a1.length; i++)
            A[i] = tanh(a1[i]);
        return A;
    }
    
    public static double tanh(double a){
        //tanh function
        double a1=Math.exp(-a);
        double a2=Math.exp(a);
        return (a2-a1)/(a2+a1);
    }
    
    public static double derivativeTanh (double a){
        //derivative for tanh
        return 1 - Math.pow(a, 2);
    }

    public static double[] derivativeTanhVector (double[]a){
        //derivative for tanh for vectors
        double[] b = new double[a.length];
        for(int i = 0; i < a.length; i++)
            b[i] = derivativeTanh(a[i]);
        return b;
    }
    
    public static double[] dotDot(double[]a1,double[]a2){
        // .dot
        double[]c;c=new double[a1.length];
        for(int i=0;i<a1.length;i++)
            c[i]=a1[i]*a2[i];
        return c;
    }
}