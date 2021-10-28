/*    
*    ForestCoverTypeTestBed.java 
*    Copyright (C) 2017 Universidade Federal do Paraná, Curitiba, Paraná, Brasil
*    @Author Paulo Ricardo Lisboa de Almeida (prlalmeida@inf.ufpr.br)
*    This program is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*    
*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*    
*    You should have received a copy of the GNU General Public License
*    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
package br.ufpr.dynse.testbed;

import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.nio.file.*;

import br.ufpr.dynse.classifier.factory.AbstractDynseFactory;
import br.ufpr.dynse.classifier.factory.RealConceptDriftDynseFactory;
import br.ufpr.dynse.constant.Constants;
import br.ufpr.dynse.core.StreamDynse;
import br.ufpr.dynse.core.UFPRLearningCurve;
import br.ufpr.dynse.evaluation.EvaluatePeriodicHeldOutTestUFPR;
import br.ufpr.dynse.evaluation.EvaluateInterleavedChunksUFPRMasked;
import br.ufpr.dynse.util.UFPRLearningCurveUtils;
import moa.streams.ArffFileStream;
import moa.tasks.StandardTaskMonitor;
import moa.tasks.TaskMonitor;

public class MaskedTestBed implements MultipleExecutionsTestbed{

	// private static final String PATH_DATASET = "datasets/covtypeNorm.arff";
	// private static final String PATH_DATASET = "G:/My Drive/UniMine/Uni/PhD/AirStream/JournalRevision/AdditionalComparisons/dynse-master/dynse/src/main/java/br/ufpr/dynse/testbed/datasets/covtypeNorm.arff";
	// private static final String PATH_DATASET = "G:/My Drive/UniMine/Uni/PhD/AirStream/JournalRevision/AdditionalComparisons/dynse-master/dynse/src/main/java/br/ufpr/dynse/testbed/datasets/3811.arff";
	private static final String PATH_DATASET = "G:/My Drive/UniMine/Uni/PhD/AirStream/JournalRevision/AdditionalComparisons/dynse-master/dynse/src/main/java/br/ufpr/dynse/testbed/datasets/stream-RangioraClean_dataset.arff";
	
	private UFPRLearningCurveUtils ufprLearningCurveUtils = new UFPRLearningCurveUtils();
	private final AbstractDynseFactory dynseFactory = new RealConceptDriftDynseFactory();

	@Override
	public void executeTests(int numExec) throws Exception{
		this.executeTestsKU(numExec);
	}
	public void executeExternalTests(int numExec, String inputFilePath, String outputFileName) throws Exception{
		this.executeTestsKE(numExec, inputFilePath, outputFileName);
	}
	public void executeTestsKE(int numberExecutions, String inputFilePath, String outputFileName) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numberExecutions);
		for(int i =0;i < numberExecutions; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluateInterleavedChunksUFPRMasked evaluator = new EvaluateInterleavedChunksUFPRMasked();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(inputFilePath);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseKE(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			// evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			int index = inputFilePath.lastIndexOf(".");
			String ext = inputFilePath.substring(0, index);
			Path inputPath = Paths.get(inputFilePath);
			Path parent = inputPath.getParent();
			String fileName = inputPath.getFileName().toString();
			String newPath = parent.resolve(outputFileName).toString();
			System.out.println(newPath);
			evaluator.dumpFileOption.setValue(newPath);
			evaluator.sampleFrequencyOption.setValue(1);
			evaluator.prepareForUse();

			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
	
	public void executeTestsKU(int numberExecutions) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numberExecutions);
		for(int i =0;i < numberExecutions; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluatePeriodicHeldOutTestUFPR evaluator = new EvaluatePeriodicHeldOutTestUFPR();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(PATH_DATASET);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseKU(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.prepareForUse();
			
			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
	
	public void executeTestsKUW(int numberExecutions) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numberExecutions);
		for(int i =0;i < numberExecutions; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluatePeriodicHeldOutTestUFPR evaluator = new EvaluatePeriodicHeldOutTestUFPR();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(PATH_DATASET);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseKUW(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.prepareForUse();
			
			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
	
	public void executeTestsLCA(int numberExecutions) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numberExecutions);
		for(int i =0;i < numberExecutions; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluatePeriodicHeldOutTestUFPR evaluator = new EvaluatePeriodicHeldOutTestUFPR();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(PATH_DATASET);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseLCA(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.prepareForUse();
			
			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
	
	public void executeTestsOLA(int numberExecutions) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numberExecutions);
		for(int i =0;i < numberExecutions; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluatePeriodicHeldOutTestUFPR evaluator = new EvaluatePeriodicHeldOutTestUFPR();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(PATH_DATASET);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseOLA(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.prepareForUse();
			
			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
	
	public void executeTestsAPriori(int numExec) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numExec);
		for(int i =0;i < numExec; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluatePeriodicHeldOutTestUFPR evaluator = new EvaluatePeriodicHeldOutTestUFPR();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(PATH_DATASET);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseAPriori(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.prepareForUse();
			
			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
	
	public void executeTestsAPosteriori(int numExec) throws Exception{
		List<UFPRLearningCurve> learningCurves = new ArrayList<UFPRLearningCurve>(numExec);
		for(int i =0;i < numExec; i++){
			TaskMonitor monitor = new StandardTaskMonitor();
			EvaluatePeriodicHeldOutTestUFPR evaluator = new EvaluatePeriodicHeldOutTestUFPR();
			
			ArffFileStream stream = new ArffFileStream();
			stream.arffFileOption.setValue(PATH_DATASET);
			
			StreamDynse streamKnoraDriftHandler = dynseFactory.createDefaultDynseAPosteriori(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.learnerOption.setCurrentObject(streamKnoraDriftHandler);
			
			StringBuilder builder = new StringBuilder();
			streamKnoraDriftHandler.getShortDescription(builder, 0);
			System.out.println("Running " + i + " " + builder);
			
			evaluator.streamOption.setCurrentObject(stream);
			evaluator.testSizeOption.setValue(Constants.NUM_INST_TEST_CLASSIFIER_FOREST);
			evaluator.sampleFrequencyOption.setValue(Constants.NUM_INST_TRAIN_CLASSIFIER_FOREST);
			evaluator.prepareForUse();
			
			UFPRLearningCurve lc = (UFPRLearningCurve)evaluator.doTask(monitor, null);
			System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(lc));
			learningCurves.add(lc);
		}
		
		UFPRLearningCurve avgResult = ufprLearningCurveUtils.averageResults(learningCurves);
		System.out.println(ufprLearningCurveUtils.strMainStatisticsMatlab(avgResult));
	}
}