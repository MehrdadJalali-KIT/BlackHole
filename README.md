<div align="center">
    <img src="BH2.png" alt="Black Hole Strategy in MOF Networks" width="400">
    <p><strong>Black Hole Strategy</strong>: A gravity-inspired graph sparsification approach for Metal-Organic Framework (MOF) networks.</p>
</div>

<h1>Black Hole Strategy for Graph Sparsification in MOF Networks</h1>

The <strong>Black Hole (BH)</strong> strategy is a novel graph sparsification technique inspired by the gravitational pull of black holes, which condense matter into highly structured forms. BH retains the most influential nodes and edges in a network using a gravity-like scoring mechanism, preserving critical connections and community structures (via the <strong>Louvain algorithm</strong>). Unlike random pruning, BH ensures stable and accurate graph representations, even under extreme sparsity, making it ideal for downstream learning tasks in sparse regimes.

<div align="center">
    <img src="Animated_BH_txt_shorter.gif" alt="Animation of Black Hole Sparsification" width="400">
</div>

<h2>About the Project</h2>

The <strong>Black Hole Strategy</strong> is implemented within the <strong>MOFGalaxyNet</strong> framework, designed for analyzing Metal-Organic Framework (MOF) networks. It leverages <strong>weighted edge importance</strong> and <strong>community detection</strong> to sparsify graphs while maintaining structural integrity, outperforming traditional methods like random pruning or edge betweenness centrality.

For the full MOFGalaxyNet code, visit:  
<a href="https://github.com/MehrdadJalali-KIT/MOFGalaxyNet">MehrdadJalali-AI/MOFGalaxyNet</a>

For more details about the project and my work, visit my personal website:  
<a href="https://www.mehrdadjalali.de">www.mehrdadjalali.de</a>

<h2>Key Features</h2>

<ul>
    <li><strong>Gravity-Inspired Sparsification</strong>: Uses a scoring mechanism to prioritize influential edges and nodes.</li>
    <li><strong>Community обслуживание</strong>: Integrates Louvain algorithm for robust community detection.</li>
    <li><strong>High Stability in Sparse Regimes</strong>: Maintains graph accuracy for downstream tasks like GraphSAGE training.</li>
    <li><strong>Comprehensive Analysis Tools</strong>: Includes modules for visualizing MOF properties (e.g., linker and metal distributions, pore-limiting diameters).</li>
</ul>

<h2>Repository Structure</h2>

<h3>Core Modules</h3>
<table>
    <tr>
        <th>File</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>main.py</code></td>
        <td>Orchestrates the pipeline: data loading, BH sparsification, GraphSAGE training.</td>
    </tr>
    <tr>
        <td><code>data_utils.py</code></td>
        <td>Loads and preprocesses data, generates RDKit fingerprints, and cleans features.</td>
    </tr>
    <tr>
        <td><code>graphsage_model.py</code></td>
        <td>Defines and trains the GraphSAGE neural network.</td>
    </tr>
    <tr>
        <td><code>bh_sparsification.py</code></td>
        <td>Implements the Black Hole sparsification algorithm.</td>
    </tr>
    <tr>
        <td><code>experiment_manager.py</code></td>
        <td>Manages experiments, logging, checkpoints, and result storage.</td>
    </tr>
    <tr>
        <td><code>sparsification_methods.py</code></td>
        <td>Provides alternative sparsification methods for comparison (e.g., random pruning).</td>
    </tr>
</table>

<h3>Analysis & Visualization Modules</h3>
<table>
    <tr>
        <th>File</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>LinkerDistribuation.py</code></td>
        <td>Visualizes the distribution of organic linkers in MOFs.</td>
    </tr>
    <tr>
        <td><code>MetalDistribuation.py</code></td>
        <td>Visualizes the distribution of metals in MOFs.</td>
    </tr>
    <tr>
        <td><code>PLDdistribuation.py</code></td>
        <td>Visualizes pore-limiting diameter (PLD) distributions in MOFs.</td>
    </tr>
    <tr>
        <td><code>analyze_sparsified_graphs_v2.py</code></td>
        <td>Analyzes properties of sparsified graphs across methods and thresholds.</td>
    </tr>
    <tr>
        <td><code>PlotNetworkParameters.py</code></td>
        <td>Plots network metrics (e.g., degree distribution, modularity).</td>
    </tr>
    <tr>
        <td><code>PlotParameters.py</code></td>
        <td>Plots sparsification metrics (accuracy, modularity, runtime).</td>
    </tr>
    <tr>
        <td><code>Plot_Redundency.py</code></td>
        <td>Visualizes redundancy and overlap metrics across sparsification thresholds.</td>
    </tr>
    <tr>
        <td><code>plot_network_metrics.py</code></td>
        <td>Generates plots for network metrics (e.g., density, clustering).</td>
    </tr>
    <tr>
        <td><code>Performance_Frugal_overall.py</code></td>
        <td>Benchmarks and summarizes performance across sparsification methods.</td>
    </tr>
</table>

<h3>Data Files</h3>
<ul>
    <li><code>MOFGalaxyNet.csv</code>: Edge list for MOF network (829,300 edges, not included).</li>
    <li><code>MOFCSD.csv</code>: Node features for MOFs (12,561 nodes, not included).</li>
    <li><code>sparsification_performance.csv</code>: Stores evaluation metrics (accuracy, modularity, runtime).</li>
    <li><code>BH.jpg</code> & <code>Animated_BH_txt_shorter.gif</code>: Visuals for README.</li>
</ul>

<h2>Installation</h2>

<h3>Prerequisites</h3>
<ul>
    <li>Python 3.9</li>
    <li>Conda</li>
    <li>Required packages: <code>pytorch==2.4.0</code>, <code>pandas</code>, <code>numpy</code>, <code>networkx</code>, <code>scikit-learn</code>, <code>rdkit>=2024.03</code>, <code>psutil</code>, <code>tqdm</code></li>
</ul>

<h3>Setup Instructions</h3>
<ol>
    <li><strong>Create a Conda environment</strong>:
        <pre><code>conda create -n bh_env python=3.9
conda activate bh_env</code></pre>
    </li>
    <li><strong>Install dependencies</strong>:
        <pre><code>conda install pytorch==2.4.0 pandas numpy networkx scikit-learn rdkit psutil -c pytorch -c conda-forge
pip install tqdm</code></pre>
    </li>
    <li><strong>Clone the repository</strong>:
        <pre><code>git clone https://github.com/MehrdadJalali-KIT/black-hole-strategy.git
cd black-hole-strategy</code></pre>
    </li>
</ol>

<p><strong>Note</strong>: For RDKit >=2024.03, update <code>data_utils.py</code> to use <code>MorganGenerator.GetFingerprintAsNumPy</code> to avoid deprecation warnings. See <a href="#troubleshooting">Troubleshooting</a> for details.</p>

<h2>Usage</h2>

<ol>
    <li><strong>Prepare data</strong>: Place <code>MOFGalaxyNet.csv</code>, <code>MOFCSD.csv</code>, <code>BH.jpg</code>, and <code>Animated_BH_txt_shorter.gif</code> in the project root.</li>
    <li><strong>Run the pipeline</strong>:
        <pre><code>conda activate bh_env
rm -rf __pycache__ *.pyc
python main.py</code></pre>
    </li>
    <li><strong>Monitor progress</strong>:
        <pre><code>tail -f bh_evaluation.log</code></pre>
    </li>
</ol>

<h3>Expected Output</h3>
<ul>
    <li><strong>Logs</strong>: Feature generation (<code>[12561, 1031]</code>), training progress (e.g., <code>Epoch X, Loss: Y, Train Accuracy: Z</code>), and test accuracy (0.6–0.8).</li>
    <li><strong>Results</strong>: Stored in <code>evaluation/threshold_0.90/method_{blackhole,random}/run_0/model_results_with_error_bars.csv</code>.</li>
    <li><strong>Run Time</strong>: ~10 minutes on a typical CPU (e.g., Apple Silicon).</li>
</ul>

<h3>Optional: Test Without Edge Weights</h3>
<p>To match previous versions, edit <code>main.py</code>:</p>
<pre><code>use_edge_weights = False</code></pre>
<p>Then run:</p>
<pre><code>python main.py</code></pre>

<h2>Troubleshooting</h2>

<ol>
    <li><strong>Check Logs</strong>:
        <pre><code>tail -f bh_evaluation.log</code></pre>
        <p>Verify: Feature shape <code>[12561, 1031]</code>, test accuracy >0.5, no feature shape mismatches.</p>
    </li>
    <li><strong>Validate Data</strong>:
        <pre><code>import pandas as pd
summary = pd.read_csv('MOFCSD.csv')
print(f"Invalid SMILES: {(summary['linker SMILES'] == 'F[Si](F)(F)(F)(F)F').sum()}")
print(f"Metals: {summary['metal'].value_counts()}")
print(f"NaNs: {summary[['Pore Limiting Diameter', 'Largest Cavity Diameter', 'Largest Free Sphere']].isna().sum()}")</code></pre>
    </li>
    <li><strong>Test Data Loading</strong>:
        <pre><code>from data_utils import load_summary_data
edges = pd.read_csv('MOFGalaxyNet.csv')
nodes = pd.concat([edges['source'], edges['target']]).unique()
features_df, summary_data = load_summary_data('MOFCSD.csv', nodes)
print(features_df.shape)  # Should be (12561, 1031)</code></pre>
    </li>
    <li><strong>RDKit Version</strong>:
        <pre><code>python -c "from rdkit import __version__; print(__version__)"</code></pre>
        <p>If <2024.03, update RDKit and modify <code>data_utils.py</code> to use <code>MorganGenerator</code>.</p>
    </li>
</ol>

<h2>Known Issues</h2>

<ul>
    <li><strong>RDKit Deprecation</strong>: Older RDKit versions (<2024.03) trigger warnings for <code>AllChem.GetMorganFingerprintAsBitVect</code>. Update RDKit and use <code>MorganGenerator</code>.</li>
    <li><strong>Invalid SMILES</strong>: <code>F[Si](F)(F)(F)(F)F</code> in <code>MOFCSD.csv</code> is replaced with <code>c1ccccc1</code> (benzene) during preprocessing.</li>
    <li><strong>Low Accuracy</strong>: Accuracy may be lower than previous versions (>0.5). Test with <code>use_edge_weights = False</code>.</li>
    <li><strong>Modularity</strong>: Black Hole partition may fall back to Louvain (modularity ~0.4075) if invalid.</li>
</ul>

<h2>Contributing</h2>

<p>Contributions are welcome! Please submit issues or pull requests to:</p>
<ul>
    <li>Fix invalid SMILES in <code>MOFCSD.csv</code>.</li>
    <li>Enhance Black Hole partition modularity in <code>bh_sparsification.py</code>.</li>
    <li>Optimize GraphSAGE for better accuracy with edge weights.</li>
</ul>

<h2>License</h2>

<p>This project is licensed under the MIT License. See <a href="LICENSE">LICENSE</a> for details.</p>

<h2>Contact</h2>

<p>For questions or collaboration, reach out via GitHub or visit my website for more information:  
<a href="https://www.mehrdadjalali.de">www.mehrdadjalali.de</a></p>
