/*    
*    EvaluateInterleavedChunksUFPRMasked.java 
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
package br.ufpr.dynse.evaluation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import br.ufpr.dynse.core.UFPRLearningCurve;
import moa.classifiers.Classifier;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.InstanceStream;
import moa.tasks.EvaluateInterleavedChunks;
import moa.tasks.TaskMonitor;
import moa.core.Utils;

public class EvaluateInterleavedChunksUFPRMasked extends EvaluateInterleavedChunks {
	
	private static final long serialVersionUID = 1L;
	
	private static final String STR_MEDICAO_CLASSIFICATIONS_CORRECT = "classifications correct (percent)";
	private static final String STR_MEDICAO_INST_CLASSIFICADAS = "classified instances";

	public Class<?> getTaskResultType() {
		return LearningCurve.class;
	}

	@Override
	protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
		Classifier learner = (Classifier) getPreparedClassOption(this.learnerOption);
		InstanceStream stream = (InstanceStream) getPreparedClassOption(this.streamOption);
		LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
		learner.setModelContext(stream.getHeader());
		int maxInstances = this.instanceLimitOption.getValue();
		int chunkSize = this.chunkSizeOption.getValue();
		long instancesProcessed = 0;
		int maxSeconds = this.timeLimitOption.getValue();
		int secondsElapsed = 0;
		double lastClassValue = 0.0;

		int instances_seen = 0;
		int right = 0;
		int wrong = 0;

		int true_label = 0;
		int predicted_label = 0;

		double memory = 0.0;


		System.out.println(this.sampleFrequencyOption.getValue());
		
		monitor.setCurrentActivity("Evaluating learner...", -1.0);
		
		UFPRLearningCurve learningCurve = new UFPRLearningCurve("example");
		File dumpFile = this.dumpFileOption.getFile();
		System.out.println(dumpFile.getName());
//		File dumpFile = new File("H:\\Test.txt");
		PrintStream immediateResultStream = null;
		if (dumpFile != null) {
			try {
				if (dumpFile.exists()) {
					immediateResultStream = new PrintStream(
							new FileOutputStream(dumpFile, true), true);
				} else {
					immediateResultStream = new PrintStream(
							new FileOutputStream(dumpFile), true);
				}
			} catch (Exception ex) {
				throw new RuntimeException(
						"Unable to open immediate result file: " + dumpFile, ex);
			}
		}
		boolean firstDump = true;
		boolean firstChunk = true;
		boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
		long sampleTestTime =0, sampleTrainTime = 0;
		double RAMHours = 0.0;
		
		while (stream.hasMoreInstances()
				&& ((maxInstances < 0) || (instancesProcessed < maxInstances))
				&& ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
			
			Instances chunkInstances = new Instances(stream.getHeader(), chunkSize);
			
			while (stream.hasMoreInstances() && chunkInstances.numInstances() < chunkSize) {
				Instance next_instance = stream.nextInstance().getData();
				chunkInstances.add(next_instance);
				if (chunkInstances.numInstances()
						% INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
					if (monitor.taskShouldAbort()) {
						return null;
					}
					
					long estimatedRemainingInstances = stream.estimatedRemainingInstances();
			
					if (maxInstances > 0) {
						long maxRemaining = maxInstances - instancesProcessed;
						if ((estimatedRemainingInstances < 0) || (maxRemaining < estimatedRemainingInstances)) {
							estimatedRemainingInstances = maxRemaining;
						}
					}
					
					monitor.setCurrentActivityFractionComplete((double) instancesProcessed/ (double) (instancesProcessed + estimatedRemainingInstances));
				}
			}
			
			////Testing
			long testStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
			boolean testeExecutado = true;
			if(!firstChunk)
			{
				for (int i=0; i< chunkInstances.numInstances(); i++) {
					////Training
					long trainStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
					Instance masked_train_instance = chunkInstances.instance(i).copy();
					int class_index = masked_train_instance.classIndex();
					double mask_val = masked_train_instance.value(class_index-1);
					if (mask_val == 0.0){
						lastClassValue = masked_train_instance.classValue();
					}else{
						masked_train_instance.setClassValue(lastClassValue);
					}
					masked_train_instance.setValue(class_index-1, 0.0);
					Example testInst = new InstanceExample((Instance) chunkInstances.instance(i));
					Example testInstMask = new InstanceExample((Instance) masked_train_instance);
					double[] prediction = learner.getVotesForInstance(testInstMask);
					int predictedClass = Utils.maxIndex(prediction);
					evaluator.addResult(testInst, prediction);
					instances_seen++;
					boolean is_correct = (int) chunkInstances.instance(i).classValue() == predictedClass;
					if (is_correct){
						right++;
					}else{
						wrong++;
					}
					double overall_accuracy = ((double) right) / ((double) right + (double) wrong);
					instancesProcessed++;
					sampleTrainTime += TimingUtils.getNanoCPUTimeOfCurrentThread() - trainStartTime;

					////Result output
					if (instancesProcessed > 1 && instancesProcessed % this.sampleFrequencyOption.getValue() == 0 && testeExecutado) {
						double avgTrainTime = TimingUtils.nanoTimeToSeconds(sampleTrainTime)/((double)this.sampleFrequencyOption.getValue()/chunkInstances.numInstances());
						double avgTestTime = TimingUtils.nanoTimeToSeconds(sampleTestTime)/((double)this.sampleFrequencyOption.getValue()/chunkInstances.numInstances());

						sampleTestTime = 0;
						sampleTrainTime = 0;

						List<Measurement> measurements = new ArrayList<Measurement>();
						measurements.add(new Measurement("example", instancesProcessed));
						measurements.add(new Measurement("sliding_window_accuracy", overall_accuracy));
						measurements.add(new Measurement("is_correct", is_correct ? 1 : 0));
						measurements.add(new Measurement("p", predictedClass));
						measurements.add(new Measurement("y", (int) chunkInstances.instance(i).classValue()));
						measurements.add(new Measurement("overall_accuracy", overall_accuracy));
						measurements.add(new Measurement(("evaluation time ("+ (preciseCPUTiming ? "cpu " : "") + "seconds)"),TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime)));
//						measurements.add(new Measurement("learning evaluation instances", instancesProcessed));
//
						Measurement[] performanceMeasurements = evaluator.getPerformanceMeasurements();
						Measurement instanciasClassificadas = new Measurement(STR_MEDICAO_INST_CLASSIFICADAS, chunkInstances.numInstances());
						Measurement taxaAcertos = performanceMeasurements[1];

						if(!taxaAcertos.getName().equals(STR_MEDICAO_CLASSIFICATIONS_CORRECT))
							throw new RuntimeException("O nome da medida de taxa de acertos não está igual a \"classified instances\","
									+ " o que pode indicar que a medida mudou de posição.");

						for (int mi = 0; mi < performanceMeasurements.length; mi++) {
							Measurement measurement = performanceMeasurements[mi];
							measurements.add(measurement);
						}

						Measurement[] modelMeasurements = learner.getModelMeasurements();
						for (Measurement measurement : modelMeasurements) {
							measurements.add(measurement);

						}
						if(instancesProcessed % 1000 == 0 || stream.estimatedRemainingInstances() < 10){
							double mem = learner.measureByteSize();
							memory = mem;
						}
						measurements.add(new Measurement("memory", memory));
						learningCurve.insertEntry(taxaAcertos, instanciasClassificadas,measurements);
//						System.out.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
//						System.out.println(learningCurve.headerToString());

						if (immediateResultStream != null) {
							if (firstDump) {
								immediateResultStream.println(learningCurve
										.headerToString());
								firstDump = false;
							}
							immediateResultStream.println(learningCurve
									.entryToString(learningCurve.numEntries() - 1));
							immediateResultStream.flush();
						}
					}
			    }
				testeExecutado = true;
			}
			else
			{
				firstChunk = false;
			}
			
			sampleTestTime += TimingUtils.getNanoCPUTimeOfCurrentThread() - testStartTime;
			

			
			for (int i=0; i< chunkInstances.numInstances(); i++) {
				Instance masked_train_instance = chunkInstances.instance(i).copy();
				int class_index = masked_train_instance.classIndex();
				// The mask is the second to last feature.
				// If the mask is 1.0, this means the true label is masked
				// and cannot be seen. To train and predict, we can use the
				// last seed ground truth value (i.e. the last pollution level.)
				// Or we can use the prediction as the ground truth.
				// The least value gets the best accuracy here so we use that.
				double mask_val = masked_train_instance.value(class_index-1);
				if (mask_val == 0.0){
					lastClassValue = masked_train_instance.classValue();
				}else{
//					Example testInstMask = new InstanceExample((Instance) masked_train_instance);
//					double[] classVotes = learner.getVotesForInstance(testInstMask);
//					int predictedClass = Utils.maxIndex(classVotes);
//					masked_train_instance.setClassValue((double) predictedClass);
					masked_train_instance.setClassValue(lastClassValue);

				}
				masked_train_instance.setValue(class_index-1, 0.0);

//				learner.trainOnInstance(chunkInstances.instance(i));
				learner.trainOnInstance(masked_train_instance);

		    }
			

			
			////Memory testing
			if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
				if (monitor.taskShouldAbort()) {
					return null;
				}
				long estimatedRemainingInstances = stream
						.estimatedRemainingInstances();
				if (maxInstances > 0) {
					long maxRemaining = maxInstances - instancesProcessed;
					if ((estimatedRemainingInstances < 0)
							|| (maxRemaining < estimatedRemainingInstances)) {
						estimatedRemainingInstances = maxRemaining;
					}
				}
				monitor
						.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
								: (double) instancesProcessed
										/ (double) (instancesProcessed + estimatedRemainingInstances));
				if (monitor.resultPreviewRequested()) {
					monitor.setLatestResultPreview(learningCurve.copy());
				}
				secondsElapsed = (int) TimingUtils
						.nanoTimeToSeconds(TimingUtils
								.getNanoCPUTimeOfCurrentThread()
								- evaluateStartTime);
			}
		}
		if (immediateResultStream != null) {
			immediateResultStream.close();
		}
		return learningCurve;
	}
}