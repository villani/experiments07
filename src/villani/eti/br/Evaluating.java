package villani.eti.br;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.TreeMap;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HMC;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Evaluating {

	/**
	 * @param id
	 * @param log
	 * @param entradas
	 */
	public static void run(String id, LogBuilder log,
			TreeMap<String, String> entradas) {

		boolean ehd = Boolean.parseBoolean(entradas.get("ehd"));
		boolean lbp = Boolean.parseBoolean(entradas.get("lbp"));
		boolean sift = Boolean.parseBoolean(entradas.get("sift"));
		boolean gabor = Boolean.parseBoolean(entradas.get("gabor"));
		boolean mlknn = Boolean.parseBoolean(entradas.get("mlknn"));
		boolean brknn = Boolean.parseBoolean(entradas.get("brknn"));
		boolean chain = Boolean.parseBoolean(entradas.get("chain"));
		boolean lp = Boolean.parseBoolean(entradas.get("lp"));
		boolean hmc_t = Boolean.parseBoolean(entradas.get("hmc_t"));
		boolean hmc_a = Boolean.parseBoolean(entradas.get("hmc_a"));
		String[] tecnicas = { "Ehd", "Lbp", "Sift", "Gabor" };
		String[] eixos = { "T", "D", "A", "B" };
		String[] classificadores = {"MLkNN", "BRkNN", "Chain", "LP", "HMC_T", "HMC_A"};

		for (String tecnica : tecnicas) {

			if (tecnica.equals("Ehd") && !ehd) continue;
			if (tecnica.equals("Lbp") && !lbp) continue;
			if (tecnica.equals("Sift") && !sift) continue;
			if (tecnica.equals("Gabor") && !gabor) continue;

			for (int i = 0; i < 10; i++) {

				for (String eixo : eixos) {

					for (String classificador : classificadores) {

						if (classificador.equals("MLkNN") && !mlknn) continue;
						if (classificador.equals("BRkNN") && !brknn) continue;
						if (classificador.equals("Chain") && !chain) continue;
						if (classificador.equals("LP") && !lp) continue;
						if (classificador.equals("HMC_T") && !hmc_t) continue;
						if (classificador.equals("HMC_A") && !hmc_a) continue;

						String nomeTreino = "Bases/" + tecnica + "-Sub" + i + "-" + eixo;

						Instances instanciasTreino = null;
						try {
							log.write(" - Desserializando instancias de treino a partir de " + nomeTreino);
							FileInputStream instanciasFIS = new FileInputStream(nomeTreino + ".bsi");
							ObjectInputStream instanciasOIS = new ObjectInputStream(instanciasFIS);
							instanciasTreino = (Instances) instanciasOIS.readObject();
							instanciasOIS.close();
							instanciasFIS.close();
						} catch (Exception e) {
							log.write(" - Falha ao desserializar instancias: " + e.getMessage());
							System.exit(0);
						}

						LabelsMetaDataImpl rotulosTreino = null;
						try {
							log.write(" - Desserializando respectiva estrutura de rótulos");
							FileInputStream rotulosFIS = new FileInputStream(nomeTreino + ".labels");
							ObjectInputStream rotulosOIS = new ObjectInputStream(rotulosFIS);
							rotulosTreino = (LabelsMetaDataImpl) rotulosOIS.readObject();
							rotulosOIS.close();
							rotulosFIS.close();
						} catch (Exception e) {
							log.write(" - Falha ao desserializar rotulos: " + e.getMessage());
							System.exit(0);
						}

						MultiLabelInstances baseTreino = null;
						try {
							log.write(" - Instanciando conjunto de treinamento multirrótulo");
							baseTreino = new MultiLabelInstances(instanciasTreino, rotulosTreino);
						} catch (InvalidDataFormatException idfe) {
							log.write(" - Erro no formato de dados ao instanciar conjunto multirrótulos: " + idfe.getMessage());
							System.exit(0);
						}

						MultiLabelLearnerBase mlLearner = null;
						log.write(" - Instanciando classificador " + classificador);
						if (classificador.equals("MLkNN")) mlLearner = new MLkNN(); // default k=10
						if (classificador.equals("BRkNN")) mlLearner = new BRkNN(); // default k=10
						if (classificador.equals("Chain")) {
							IBk kNN = new IBk(10);
							mlLearner = new ClassifierChain(kNN);
						}
						if (classificador.equals("LP")) {
							IBk kNN = new IBk(10);
							mlLearner = new LabelPowerset(kNN);
						}
						if (classificador.equals("HMC_T")) mlLearner = new HMC();
						if (classificador.equals("HMC_A")) {
							mlLearner = new HMC(new MLkNN());
						}

						try {
							log.write(" - Construindo modelo do " + classificador + " a partir do conjunto de treinamento " + nomeTreino);
							mlLearner.build(baseTreino);
						} catch (Exception e) {
							log.write(" - Falha ao construir o modelo do classificador: " + e.getMessage());
							System.exit(0);
						}

						log.write(" - Instanciando avaliador");
						Evaluator avaliador = new Evaluator();

						log.write(" - Instanciando lista de medidas");
						ArrayList<Measure> medidas = new ArrayList<Measure>();
						medidas.add(new HammingLoss());
						medidas.add(new SubsetAccuracy());
						medidas.add(new ExampleBasedPrecision());
						medidas.add(new ExampleBasedRecall());
						medidas.add(new ExampleBasedFMeasure());
						medidas.add(new ExampleBasedAccuracy());
						medidas.add(new ExampleBasedSpecificity());
						int numOfLabels = baseTreino.getNumLabels();
						medidas.add(new MicroPrecision(numOfLabels));
						medidas.add(new MicroRecall(numOfLabels));
						medidas.add(new MicroFMeasure(numOfLabels));
						medidas.add(new AveragePrecision());
						medidas.add(new Coverage());
						medidas.add(new OneError());
						medidas.add(new IsError());
						medidas.add(new ErrorSetSize());
						medidas.add(new RankingLoss());

						for (int j = 1; j < 10; j++) {
							
							if( i == j ) continue;

							String nomeTeste = "Bases/" + tecnica + "-Sub" + j + "-" + eixo;

							Instances instanciasTeste = null;
							try {
								log.write(" - Desserializando instancias de teste a partir de " + nomeTeste);
								FileInputStream instanciasFIS = new FileInputStream(nomeTeste + ".bsi");
								ObjectInputStream instanciasOIS = new ObjectInputStream(instanciasFIS);
								instanciasTreino = (Instances) instanciasOIS.readObject();
								instanciasOIS.close();
								instanciasFIS.close();
							} catch (Exception e) {
								log.write(" - Falha ao desserializar instancias: " + e.getMessage());
								System.exit(0);
							}

							LabelsMetaDataImpl rotulosTeste = null;
							try {
								log.write(" - Desserializando respectiva estrutura de rótulos");
								FileInputStream rotulosFIS = new FileInputStream(nomeTeste + ".labels");
								ObjectInputStream rotulosOIS = new ObjectInputStream(rotulosFIS);
								rotulosTreino = (LabelsMetaDataImpl) rotulosOIS.readObject();
								rotulosOIS.close();
								rotulosFIS.close();
							} catch (Exception e) {
								log.write(" - Falha ao desserializar rotulos: " + e.getMessage());
								System.exit(0);
							}

							MultiLabelInstances baseTeste = null;
							try {
								log.write(" - Instanciando conjunto de teste multirrótulo");
								baseTeste = new MultiLabelInstances(instanciasTeste, rotulosTeste);
							} catch (InvalidDataFormatException idfe) {
								log.write(" - Erro no formato de dados ao instanciar conjunto multirrótulos: " + idfe.getMessage());
								System.exit(0);
							}

							log.write(" - Avaliando o modelo gerado pelo classificador " + classificador);
							Evaluation avaliacao = null;
							try {
								avaliacao = avaliador.evaluate(mlLearner, baseTeste, medidas);
							} catch (IllegalArgumentException iae) {
								log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
								System.exit(0);
							} catch (Exception e) {
								log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
								System.exit(0);
							}

							log.write(" - Salvando resultado da avaliação");
							File resultado = new File(id + classificador + "-" + tecnica + "-" + eixo + "-Treino" + i + "-Teste" + j + ".csv");
							try {
								FileWriter escritor = new FileWriter(resultado);
								escritor.write(avaliacao.toString());
								escritor.close();
							} catch (IOException ioe) {
								log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
								System.exit(0);
							}

						}

					}

				}

			}

		}

	}
}