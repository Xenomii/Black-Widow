from PyQt6.QtWidgets import *
from PyQt6.QtGui import QIcon, QAction, QTextCursor
from pathlib import Path
from modules import preprocess
from core import ml_core as ml
from constants import dataframe_constants as dfc
from widgets.table_model import TableModel
from widgets.graph_canvas import GraphCanvas
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import widgets.textbox_results as tb
import seaborn as sns

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setting window title and icon
        self.setWindowTitle("Black Widow")
        self.setWindowIcon(QIcon("images/logo.png"))

        # Set up functions
        self._createActions()
        self._createMenuBar()
        self._createWidgets()
        self._setLayout()

        # Show the main window
        self.show()

    # === START OF SETUP FUNCTIONS === #

    def _createActions(self):
        # === File Menu Actions === #

        # Import File Action
        self.importFileAction = QAction("&Import", self)
        self.importFileAction.setShortcut("Ctrl+I")
        self.importFileAction.setStatusTip("Import Dataset")
        self.importFileAction.triggered.connect(self.import_file)

        # Parse File Action
        self.parseFileAction = QAction("&Parse", self)
        self.parseFileAction.setShortcut("Ctrl+P")
        self.parseFileAction.setStatusTip("Parse Dataset")
        self.parseFileAction.triggered.connect(self.parse_file)

        # Exit App Action
        self.exitAppAction = QAction("&Exit", self)
        self.exitAppAction.setShortcut("Ctrl+Q")
        self.exitAppAction.setStatusTip("Exit Application")
        self.exitAppAction.triggered.connect(self.exit_app)

    def _createMenuBar(self):
        menuBar = self.menuBar()

        # Populate File Menu onto the Menu bar
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.importFileAction)
        fileMenu.addAction(self.parseFileAction)
        fileMenu.addAction(self.exitAppAction)

    def _createWidgets(self):
        # Create main widget
        #   - This widget will encompass all the other widgets
        self.mainWidget = QWidget()

        # Create sub widget
        #   - This widget will encompass the tab widget and graphs
        self.subWidget = QWidget()

        # Create graph widget
        #   - This widget will encompass the different graph figures
        #   - Right now, IDK how to combine the graph and toolbar into this widget
        self.graphWidget = QWidget()

        # Create progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(50, 100, 250, 30)
        self.progressBar.setValue(0)

        # Create table widget
        self.table = QTableView()
        self.tableLabel = QLabel("Table Data")

        # Create textbox widgets
        #   - These textboxes should display the text outputs of the machine learning algorithms
        self.textbox_ml_1 = QTextEdit()
        self.textbox_ml_2 = QTextEdit()
        self.textbox_ml_3 = QTextEdit()
        self.textbox_ml_4 = QTextEdit()
        # Set the textboxes to read only
        self.textbox_ml_1.setReadOnly(True)
        self.textbox_ml_2.setReadOnly(True)
        self.textbox_ml_3.setReadOnly(True)
        self.textbox_ml_4.setReadOnly(True)

        # Create tab widget
        #   - This tab widget will hold all the textboxes together
        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(self.textbox_ml_1, "Decision Tree")
        self.tabWidget.addTab(self.textbox_ml_2, "K Nearest Neighbours")
        self.tabWidget.addTab(self.textbox_ml_3, "K-Means Clustering")
        self.tabWidget.addTab(self.textbox_ml_4, "DBSCAN Clustering")
        self.tabWidget.tabBarClicked.connect(self.tab_clicked)
        self.tabWidget.setFixedWidth(620)

        # Create graph placeholder
        #   - Ideally it should be located at near the function that runs the Machine Learning Algorithms
        self.graph_frames_list = []
        for i in range(3):
            self.graph_frames_list.append(self._createEmptyGraphs())

        # Create import check variable
        self.fileImported = None

    def _createEmptyGraphs(self):
        return GraphCanvas(self)

    def _createPrecisionRecall(self, recall_scores, precision_scores, y_test, y_pred):
        from sklearn.metrics import precision_recall_curve
        graph = GraphCanvas(self)

        # retrieve just the probabilities for the positive class
        pos_probs = y_pred[:, 1]

        # calculate the no skill line as the proportion of the positive class
        no_skill = len(y_test[y_test == 1]) / len(y_test)

        # plot the no skill precision-recall curve
        graph.axes.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # calculate model precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, pos_probs)
        # plot the model precision-recall curve
        graph.axes.plot(recall, precision, marker='.', label='KNN')
        # axis labels
        graph.axes.set_xlabel('Recall')
        graph.axes.set_ylabel('Precision')
        # show the legend
        graph.axes.legend()

        return graph

    def _createHeatMap(self, dataframe=None):
        graph = GraphCanvas(self)
        dataframe_sum = np.sum(dataframe, axis=1, keepdims=True)
        dataframe_perc = dataframe/dataframe_sum.astype('float') * 100
        sns.heatmap(dataframe_perc, annot=True, fmt=".2f", cmap=sns.cubehelix_palette(as_cmap=True) ,ax=graph.axes)
        return graph

    def _createCorrelationMatrix(self, dataframe=None):
        graph = GraphCanvas(self)

        dataframe = dataframe.drop('severity', axis=1)

        corr = dataframe.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        ticks_label = ['response', 'agent', 'HTTP', 'URL']
        sns.heatmap(
            corr, mask=mask, cmap=cmap, square=True, 
            cbar_kws={"shrink": .5}, annot=True, fmt=".1f", 
            xticklabels=ticks_label, yticklabels=ticks_label ,ax=graph.axes)
        
        graph.axes.tick_params(axis='both', which='major', labelsize=8)
        graph.axes.tick_params(axis='x', which='major', labelrotation=0)
        graph.fig.set_layout_engine('tight')

        return graph

    def _createScatter(self, x:str, y:str, category_class:str ,dataframe=None):
        graph = GraphCanvas(self)
        sns.scatterplot(data=dataframe, x=x, y=y, hue=category_class, style=category_class, ax=graph.axes)
        graph.fig.set_layout_engine('tight')
        return graph

    def _createDBScatterGraphs(self, X, labels, n_clusters, n_noise, core_samples_mask):
        graph = GraphCanvas(self)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            graph.axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10)
            xy = X[class_member_mask & ~core_samples_mask]
            graph.axes.plot(xy[:, 0], xy[:, 1], 'x', c='red', markersize=14)

        graph.axes.set_title('Estimated number of clusters: %d' % n_clusters)

        return graph

    def _createElbowGraphs(self, wcss):
        graph = GraphCanvas(self)

        # Plotting the results onto a line graph, allowing us to observe 'The elbow'
        graph.axes.plot(range(1, 11), wcss)
        graph.axes.set_title('The elbow method')
        graph.axes.set_xlabel('Number of clusters')
        graph.axes.set_ylabel('WCSS')  # within cluster sum of squares
        graph.fig.set_layout_engine('tight')

        return graph

    def _createKMScatterGraphs(self, y_kmeans, x, kmeans, title):
        graph = GraphCanvas(self)

        # filter rows of original data
        response_code = x[y_kmeans == 0]
        useragent = x[y_kmeans == 1]
        request = x[y_kmeans == 2]
        url = x[y_kmeans == 3]
        #
        # # Plotting the results
        graph.axes.scatter(response_code[:, 0], response_code[:, 1], color='purple', label='0')
        graph.axes.scatter(useragent[:, 0], useragent[:, 1], color='orange', label='1')
        graph.axes.scatter(request[:, 0], request[:, 1], color='green', label='2')
        graph.axes.scatter(url[:, 0], url[:, 1], color='blue', label='3')
        
        # Plotting the centroids of the clusters
        graph.axes.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')

        graph.axes.set_title(title)
        graph.axes.legend()
        graph.axes.figure.set_size_inches(3, 3)

        return graph

    def _createBarGraphs(self, columnName:str, dataframe=None):
        graph = GraphCanvas(self)
        sns.histplot(data=dataframe, x=columnName, kde=True, ax=graph.axes)
        graph.fig.set_layout_engine('tight')
        return graph

    def _setLayout(self):
        # Main Layout
        #   - The main layout will be a vertical layout
        #   - It is set to the main widget
        self.mainLayout = QVBoxLayout()

        # Sub Layout
        #   - The sub layout will be a horizontal layout
        #   - It is set to the sub widget
        self.subLayout = QHBoxLayout()

        # Graph Layout
        #   - The graph layout will be a grid layout
        #   - It is set to the graph widget
        self.graphLayout = QGridLayout()

        # Add widgets to graph widget first
        #   - Placeholder method to fill in the gaps for the graph figures
        self.display_graphs(self.graph_frames_list)

        # Add widgets to sub layout second
        self.subLayout.addWidget(self.tabWidget)
        self.subLayout.addLayout(self.graphLayout)

        # Add widgets to main layout
        self.mainLayout.addWidget(self.tableLabel)
        self.mainLayout.addWidget(self.table)
        self.mainLayout.addWidget(self.progressBar)
        # Note: The sub widget holds the tab widget and graph widget
        self.mainLayout.addLayout(self.subLayout)
        self.mainLayout.addWidget(self.subWidget)

        # Set main layout to main widget
        self.mainWidget.setLayout(self.mainLayout)

        self.setCentralWidget(self.mainWidget)

    # === END OF SETUP FUNCTIONS === #

    # === START OF MISC FUNCTIONS === #
    # Functions that might be used by event functions

    # Function to easily create message boxes
    def message_box(self, title="Error", message="Something went wrong...", icon=QMessageBox.Icon.Critical):
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setIcon(icon)
        dialog.exec()

    # Function to display graph onto the application based on given array of graph figures
    def display_graphs(self, graphList):
        count = 0
        for i, graph_frame in enumerate(graphList):
            if len(self.graph_frames_list) <= 3:
                if graph_frame is not None:
                    self.graphLayout.addWidget(graph_frame, 0, i)
            elif len(graphList) > 3:
                if i < 3:
                    self.graphLayout.addWidget(graph_frame, 0, i)
                else:
                    self.graphLayout.addWidget(graph_frame, 1, count)
                    count += 1

    # Function to print the results from executing machine learning algorithm
    def print_results(self, index=None, message=None):
        # If index is not given, display a random error message box
        if index is None:
            self.message_box()
            return
        elif index == 0:
            text = self.textbox_ml_1
        elif index == 1:
            text = self.textbox_ml_2
        elif index == 2:
            text = self.textbox_ml_3
        elif index == 3:
            text = self.textbox_ml_4
        
        text.clear()
        text.append(message)

        # Response Code
        actual_res = self.labelled_data.drop_duplicates(subset = ['response_code'])
        actual_res = actual_res['response_code']
        text.append("\nActual Response Code:\n" + actual_res.to_string(index=False) + "\n")

        encoded_res = self.processed_data.drop_duplicates(subset= ['response_code'])
        encoded_res = encoded_res['response_code']
        text.append("Encoded Response Code:\n" + encoded_res.to_string(index=False) + "\n")

        # User Agent
        actual_res = self.labelled_data.drop_duplicates(subset=['useragent'])
        actual_res = actual_res['useragent']
        text.append("\nActual User Agent:\n" + actual_res.to_string(index=False) + "\n")

        encoded_res = self.processed_data.drop_duplicates(subset=['useragent'])
        encoded_res = encoded_res['useragent']
        text.append("Encoded User Agent:\n" + encoded_res.to_string(index=False) + "\n")

        # HTTP Request Type
        actual_res = self.labelled_data.drop_duplicates(subset=['http_request_type'])
        actual_res = actual_res['http_request_type']
        text.append("\nActual HTTP Request Type:\n" + actual_res.to_string(index=False) + "\n")

        encoded_res = self.processed_data.drop_duplicates(subset=['http_request_type'])
        encoded_res = encoded_res['http_request_type']
        text.append("Encoded HTTP Request Type:\n" + encoded_res.to_string(index=False) + "\n")

        # URL Path
        actual_res = self.labelled_data.drop_duplicates(subset=['url_path'])
        actual_res = actual_res['url_path']
        text.append("\nActual URL Path:\n" + actual_res.to_string(index=False) + "\n")

        encoded_res = self.processed_data.drop_duplicates(subset=['url_path'])
        encoded_res = encoded_res['url_path']
        text.append("Encoded URL Path:\n" + encoded_res.to_string(index=False) + "\n")

        # We only needed one move cursor
        text.moveCursor(QTextCursor.MoveOperation.Start)

    # === END OF MISC FUNCTIONS === #

    # === START OF EVENT FUNCTIONS === #
    # When something gets clicked, what happens?

    # Executed when user clicks on Import
    def import_file(self):
        # Open file dialog at home directory
        home_dir = str(Path.home())
        file_name = QFileDialog.getOpenFileName(self, "Import Dataset", home_dir, "CSV Files (*.csv)")

        # If file has been successfully selected, proceed to preprocess...
        if file_name[0]:
            # Variable to determine whether the import action has been executed
            self.fileImported = True

            # Reset progress bar value to 0
            self.progressBar.setValue(0)

            # Get Pandas dataframe
            self.dataframe = pd.read_csv(file_name[0], names=dfc.parser_header, low_memory=False)

            # Process the data
            #   - Involves deleting useless columns
            #   - Labeling the data (i.e., establishing severity levels)
            #   - Encoding the data (i.e., turning string values into numerical values)
            self.processed_data = preprocess.delete_columns(dataframe=self.dataframe)
            self.progressBar.setValue(25)
            self.processed_data = preprocess.label_data(self.processed_data, progress_bar=self.progressBar)
            self.processed_data = preprocess.sample(self.processed_data)
            self.labelled_data = self.processed_data.copy()
            self.progressBar.setValue(50)
            self.processed_data = preprocess.label_encoder(self.processed_data)
            self.encoded_data = self.processed_data.copy()
            self.progressBar.setValue(75)

            # Structure the dataframe so that it can be outputted to the QTableView object
            tm = TableModel(self.labelled_data)
            self.table.setModel(tm)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.progressBar.setValue(100)

    # Executed when user clicks on Parse
    def parse_file(self):
        # Open file dialog at home directory
        home_dir = str(Path.home())
        file_name = QFileDialog.getOpenFileName(self, "Parse Log File", home_dir, "Log Files (*.log)")
        # Location of the output file after parsing
        csv_file_name = file_name[0] + ".csv"

        # If file has been successfully imported, proceed with parsing...
        if file_name[0]:
            try:
                # Execute the Python parser code
                #   - If the command fails, throw an Exception
                parse_command = "python src/modules/parser.py -i " + file_name[0] + " -o " + csv_file_name
                self.progressBar.setValue(0)
                if os.system(parse_command) != 0:
                    raise Exception("Something went wrong with parser...")
                else:
                    # Display a message box to indicate where the file has been outputted to
                    output_message = "Output file is located at " + csv_file_name
                    self.progressBar.setValue(100)
                    self.message_box(title="Success", message=output_message, icon=QMessageBox.Icon.Information)
                # self.final_parse(csv_file_name)
            except Exception as e:
                self.message_box()
                print(e)

    # Executed when any of the tabs are clicked on
    def tab_clicked(self, index):
        # Check whether the file has been imported first
        if self.fileImported and self.dataframe is not None:
            # Identify which tab has been clicked
            if index == 0:
                # Run the machine learning algorithm
                self.progressBar.setValue(0)
                accuracy, confusion, X_test, y_test, y_pred, TP, FP, FN, TN, Precision, Recall, confusion, y_scores = ml.run_decision(self.processed_data, self.progressBar)
                self.progressBar.setValue(50)

                # Remove all items in the array holding all the graph figures
                self.graph_frames_list.clear()

                # Clear the layout from pre-existing graph figures
                while self.graphLayout.count():
                    item = self.graphLayout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()

                freq_severity = self.processed_data['severity'].value_counts().sort_index()
                ind = np.arange(4)

                # Create the new graphs and append them to the array
                self.graph_frames_list.append(self._createCorrelationMatrix(dataframe=self.processed_data))
                self.graph_frames_list.append(self._createHeatMap(dataframe=confusion))
                self.graph_frames_list.append(self._createBarGraphs(columnName="response_code", dataframe=self.labelled_data))
                self.graph_frames_list.append(self._createScatter(x="useragent", y="response_code", category_class="severity" ,dataframe=self.processed_data))
                self.graph_frames_list.append(self._createScatter(x="response_code", y="url_path", category_class="severity" ,dataframe=self.processed_data))
                self.graph_frames_list.append(self._createScatter(x="url_path", y="http_request_type", category_class="severity" ,dataframe=self.processed_data))
                self.progressBar.setValue(75)

                # Print the results of the machine learning algorithm onto the corresponding textbox
                results = tb.supervised_algo_results(accuracy=accuracy, confusion=confusion, TP=TP, FP=FP, FN=FN, TN=TN, Precision=Precision, Recall=Recall, Algo="Decision Tree", y_test=y_test, y_pred=y_pred)
                self.print_results(index, message=results)
                # Display the graphs by providing the array
                self.display_graphs(self.graph_frames_list)
                self.progressBar.setValue(100)

            elif index == 1:
                # Run the machine learning algorithm
                self.progressBar.setValue(0)
                accuracy, confusion, X_test, y_test, y_pred, TP, FP, FN, TN, Precision, Recall, Accuracy, confusion, y_scores, original_x, original_y = ml.run_knn(self.processed_data, self.progressBar)
                self.progressBar.setValue(50)

                # Remove all items in the array holding all the graph figures
                self.graph_frames_list.clear()

                # Clear the layout from pre-existing graph figures
                while self.graphLayout.count():
                    item = self.graphLayout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()

                freq_severity = self.processed_data['severity'].value_counts().sort_index()
                ind = np.arange(4)

                # Create the new graphs and append them to the array
                self.graph_frames_list.append(self._createCorrelationMatrix(self.processed_data))
                self.graph_frames_list.append(self._createHeatMap(dataframe=confusion))
                self.graph_frames_list.append(self._createBarGraphs(columnName="response_code", dataframe=self.labelled_data))
                self.graph_frames_list.append(self._createScatter(x="useragent", y="response_code", category_class="severity" ,dataframe=self.processed_data))
                self.graph_frames_list.append(self._createScatter(x="response_code", y="url_path", category_class="severity" ,dataframe=self.processed_data))
                self.graph_frames_list.append(self._createScatter(x="url_path", y="http_request_type", category_class="severity" ,dataframe=self.processed_data))
                self.progressBar.setValue(75)

                # Print the results of the machine learning algorithm onto the corresponding textbox
                results = tb.supervised_algo_results(accuracy=accuracy, confusion=confusion, TP=TP, FP=FP, FN=FN, TN=TN, Precision=Precision, Recall=Recall, Algo="K-Nearest Neighbours", y_test=y_test, y_pred=y_pred)
                self.print_results(index, message=results)

                # Display the graphs by providing the array
                self.display_graphs(self.graph_frames_list)
                self.progressBar.setValue(100)

            elif index == 2:
                # Run the machine learning algo
                self.progressBar.setValue(0)
                y_kmeans, x, kmeans, wcss, accuracy = ml.run_kmeans(self.processed_data, self.progressBar)
                self.progressBar.setValue(50)

                # Remove all items in the array holding all the graph figures
                self.graph_frames_list.clear()

                # Clear the layout from pre-existing graph figures
                while self.graphLayout.count():
                    item = self.graphLayout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()

                # Create the new graphs and append them to the array
                self.graph_frames_list.append(self._createElbowGraphs(wcss))
                self.graph_frames_list.append(self._createKMScatterGraphs(y_kmeans, x, kmeans, title="K-Means Clustering (K = 5)"))
                self.progressBar.setValue(75)

                # Print the results of the machine learning algorithm onto the corresponding textbox
                results = tb.kmeans_results(pred_cluster=y_kmeans, accuracy=accuracy)
                self.print_results(index, message=results)

                # Display the graphs by providing the array
                self.display_graphs(self.graph_frames_list)
                self.progressBar.setValue(100)

            elif index == 3:
                # Run the machine learning algo
                self.progressBar.setValue(0)
                X, labels, n_clusters, n_noise, core_samples_mask = ml.run_dbscan(self.processed_data, self.progressBar)
                self.progressBar.setValue(50)

                # Remove all items in the array holding all the graph figures
                self.graph_frames_list.clear()

                # Clear the layout from pre-existing graph figures
                while self.graphLayout.count():
                    item = self.graphLayout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()

                # Create the new graphs and append them to the array
                self.graph_frames_list.append(self._createDBScatterGraphs(X, labels, n_clusters, n_noise, core_samples_mask))
                self.progressBar.setValue(75)

                # Print the results of the machine learning algorithm onto the corresponding textbox
                results = tb.dbscan_results(n_clusters=n_clusters, n_noise=n_noise)
                self.print_results(index, message=results)

                # Display the graphs by providing the array
                self.display_graphs(self.graph_frames_list)
                self.progressBar.setValue(100)
            else:
                text = None
        else:
            # If the file has not been imported yet, then display error message box
            self.message_box(message="You must import a file first!", icon=QMessageBox.Icon.Information)

    # Executed when user clicks on Exit
    def exit_app(self):
        QApplication.quit()

    # === END OF EVENT FUNCTIONS === #
