package kr.kaist.sentence.embedding.structure;

public class MultiThread {
	public double[] derivativeOutputU;
	
	public double[] derivativeBias;
	public double[][] derivativeWeightMatrix;
	
    public double[] derivativeDocumentBias;
	public double[][] derivativeDocumentWeightMatrix;

    public MultiThread(int dimension){
    	derivativeOutputU = new double[dimension];
    	
    	derivativeBias = new double[dimension];
    	derivativeWeightMatrix = new double[dimension][dimension * 2];
    	
    	derivativeDocumentBias = new double[dimension];
    	derivativeDocumentWeightMatrix = new double[dimension][dimension * 2];
    }
}
