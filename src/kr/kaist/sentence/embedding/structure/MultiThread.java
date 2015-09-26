package kr.kaist.sentence.embedding.structure;

public class MultiThread {
    public double[] derivativeBias;
	public double[][] derivativeWeightMatrix;

    public MultiThread(int dimension){
    	derivativeBias = new double[dimension];
    	derivativeWeightMatrix = new double[dimension][dimension * 2];
    }
}
