# ============================================================

# Train_Feature_Adjacent_Matrix_Final.py

#

# Excel format

# ------------------------------------------------------------

# Column A              : file name / case name

# Column B ~ blank col  : feature matrix + label column

#                         - last column before blank col = label ## Engineer's FEM intent should be reflected at this label manually

# After blank col       : adjacency matrix

# Blank row             : separator between solid bodies / graph cases

# ============================================================

import logging

from datetime import datetime

import sys

import os

import random

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from torch_geometric.nn import GCNConv

from sklearn.metrics import confusion_matrix

import subprocess

# ============================================================

# 1. CONFIG

# ============================================================

EXCEL_FILE = r"TestDataResult_Excel\cantilever.xlsx"  # Change the target geometry name before running.

# Train_Everything contains all five original geometry families.

# Specify the path if a separate validation Excel file is available.

# Otherwise, set it to None to automatically split the dataset.

VALIDATION_EXCEL_FILE = None  # Example: r"TestDataResult_Excel\cantilever.xlsx"

NUM_CLASSES = 3

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 1000

# TRAIN_RATIO = 0.8

RANDOM_SEED = 42

TRAIN_RATIO = 0.6

VALID_RATIO = 0.2

TEST_RATIO = 0.2

import torch

import torch.nn as nn

import torch.nn.functional as F

class MLPBaseline(nn.Module):

    def __init__(

        self,

        input_dim,

        hidden_dim,

        num_classes,

        dropout=0.0

    ):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, 32)

        self.fc2 = nn.Linear(32, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None):

        """

        The edge_index argument is accepted but not used.

        Each face is therefore classified independently.

        """

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.classifier(x)

        return x

def setup_experiment_log(experiment_name="experiment"):

    """

    Create a new log file for each run and record the following:

    1. Execution timestamp

    2. Python script path

    3. Experiment configuration

    4. All subsequent print() output

    5. Error messages (stderr)

    Output is written to the console and the .log file simultaneously.

    """

    # ---------------------------------------------------------

    # Internal class that redirects print output to both the console and log file

    # ---------------------------------------------------------

    class TeeStream:

        def __init__(

            self,

            original_stream,

            logger,

            log_level=logging.INFO

        ):

            self.original_stream = original_stream

            self.logger = logger

            self.log_level = log_level

            self.buffer = ""

        def write(self, message):

            # Preserve the original console output

            self.original_stream.write(message)

            self.original_stream.flush()

            # Handle print() output delivered in multiple fragments

            self.buffer += message

            while "\n" in self.buffer:

                line, self.buffer = self.buffer.split("\n", 1)

                if line.strip():

                    self.logger.log(

                        self.log_level,

                        line.rstrip()

                    )

        def flush(self):

            self.original_stream.flush()

            # Log any remaining buffered content

            if self.buffer.strip():

                self.logger.log(

                    self.log_level,

                    self.buffer.rstrip()

                )

                self.buffer = ""

    # ---------------------------------------------------------

    # Create the logs directory and log filename

    # ---------------------------------------------------------

    script_dir = os.path.dirname(

        os.path.abspath(__file__)

    )

    log_dir = os.path.join(

        script_dir,

        "logs"

    )

    os.makedirs(

        log_dir,

        exist_ok=True

    )

    timestamp = datetime.now().strftime(

        "%Y%m%d_%H%M%S"

    )

    log_name = (

        f"{timestamp}_{experiment_name}.log"

    )

    log_path = os.path.join(

        log_dir,

        log_name

    )

    # ---------------------------------------------------------

    # Create the logger

    # ---------------------------------------------------------

    logger = logging.getLogger(

        f"experiment_logger_{timestamp}"

    )

    logger.setLevel(logging.INFO)

    logger.propagate = False

    # Prevent duplicate handlers

    logger.handlers.clear()

    file_handler = logging.FileHandler(

        log_path,

        mode="w",

        encoding="utf-8"

    )

    file_handler.setFormatter(

        logging.Formatter("%(message)s")

    )

    logger.addHandler(file_handler)

    # ---------------------------------------------------------

    # Record the configuration

    # ---------------------------------------------------------

    logger.info("=" * 70)

    logger.info("EXPERIMENT CONFIGURATION")

    logger.info("=" * 70)

    logger.info(

        f"Run time              : "

        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    )

    logger.info(

        f"Experiment name       : {experiment_name}"

    )

    logger.info(

        f"Python file           : "

        f"{os.path.abspath(__file__)}"

    )

    logger.info(

        f"Excel file            : {EXCEL_FILE}"

    )

    logger.info(

        f"Validation Excel file : "

        f"{VALIDATION_EXCEL_FILE}"

    )

    logger.info(

        f"Number of classes     : {NUM_CLASSES}"

    )

    logger.info(

        f"Batch size            : {BATCH_SIZE}"

    )

    logger.info(

        f"Learning rate         : {LEARNING_RATE}"

    )

    logger.info(

        f"Epochs                : {EPOCHS}"

    )

    logger.info(

        f"Random seed           : {RANDOM_SEED}"

    )

    logger.info(

        f"Train ratio           : {TRAIN_RATIO}"

    )

    logger.info(

        f"Validation ratio      : {VALID_RATIO}"

    )

    logger.info(

        f"Test ratio            : {TEST_RATIO}"

    )

    logger.info("=" * 70)

    logger.info("")

    # ---------------------------------------------------------

    # Redirect all subsequent print and error output to the log

    # ---------------------------------------------------------

    sys.stdout = TeeStream(

        original_stream=sys.__stdout__,

        logger=logger,

        log_level=logging.INFO

    )

    sys.stderr = TeeStream(

        original_stream=sys.__stderr__,

        logger=logger,

        log_level=logging.ERROR

    )

    print("Log file:", log_path)

    print()

    return log_path

# ============================================================

# 2. SEED

# ============================================================

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

# ============================================================

# 3. adjacency matrix -> edge_index

# ============================================================

def adjacency_to_edge_index(adj):

    edge_index = []

    for i in range(len(adj)):

        for j in range(len(adj[i])):

            value = adj[i][j]

            # Handle NaN values

            if pd.isna(value):

                value = 0

            if int(value) == 1:

                edge_index.append([i, j])

    edge_index = torch.tensor(

        edge_index,

        dtype=torch.long

    )

    edge_index = edge_index.t().contiguous()

    return edge_index

# ============================================================

# 4. one solid body block -> PyG Data

# ============================================================

def process_one_case(block_df, case_index):

    block_df = block_df.reset_index(drop=True)

    # --------------------------------------------------------

    # Locate the blank separator column

    # From column B to the column before the blank separator: feature matrix

    # Columns after the blank separator: adjacency matrix

    # --------------------------------------------------------

    # =========================================================

    # column settings

    # =========================================================

    FEATURE_START_COL = 1

    # Exclusive end index for features and label

    FEATURE_END_COL = 13

    # Starting index of the adjacency matrix

    ADJ_START_COL = 14

    # =========================================================

    # feature matrix

    # =========================================================

    feature_df = block_df.iloc[

        :,

        FEATURE_START_COL:FEATURE_END_COL

    ]

    # The last feature column contains the label

    x_df = feature_df.iloc[:, :-1]

    y_df = feature_df.iloc[:, -1]

    x = torch.tensor(

        x_df.values,

        dtype=torch.float

    )

    y = torch.tensor(

        y_df.values,

        dtype=torch.long

    )

    # =========================================================

    # adjacency matrix

    # =========================================================

    num_nodes = len(x_df)

    available_adj_cols = max(0, block_df.shape[1] - ADJ_START_COL)

    read_adj_cols = min(num_nodes, available_adj_cols)

    adj_df = block_df.iloc[

        :num_nodes,

        ADJ_START_COL:ADJ_START_COL + read_adj_cols,

    ]

    raw_adj = (

        adj_df.apply(pd.to_numeric, errors="coerce")

        .fillna(0)

        .to_numpy(dtype=int)

    )

    # Initialize a square matrix

    adj = np.zeros((num_nodes, num_nodes), dtype=int)

    # Copy the available columns

    adj[:, :read_adj_cols] = raw_adj

    # Restore missing columns using matrix symmetry

    adj = np.maximum(adj, adj.T)

    # Remove self-loops

    np.fill_diagonal(adj, 0)

    edge_index = adjacency_to_edge_index(adj)

    data = Data(

        x=x,

        edge_index=edge_index,

        y=y

    )

    return data

# ============================================================

# 5. Excel -> Dataset

# ============================================================

def load_dataset_from_excel(excel_file):

    print()

    print("Loading:", excel_file)

    df = pd.read_excel(

        excel_file,

        header=None

    )

    print("Excel shape:", df.shape)

    dataset = []

    current_rows = []

    case_index = 0

    for row_index in range(len(df)):

        row = df.iloc[row_index]

        # An empty row marks the end of one solid-body graph case

        if row.isna().all():

            if len(current_rows) > 0:

                block_df = pd.DataFrame(current_rows)

                data = process_one_case(block_df, case_index)

                dataset.append(data)

                current_rows = []

                case_index += 1

        else:

            current_rows.append(row)

    # Process the final block

    if len(current_rows) > 0:

        block_df = pd.DataFrame(current_rows)

        data = process_one_case(block_df, case_index)

        dataset.append(data)

    print("Dataset size:", len(dataset))

    if len(dataset) == 0:

        raise ValueError("No graph case was loaded from Excel.")

    return dataset

# ============================================================

# 6. Train / Validation split

# ============================================================

def split_dataset(

    dataset,

    train_ratio,

    valid_ratio,

    test_ratio

):

    indices = list(range(len(dataset)))

    random.shuffle(indices)

    total = len(indices)

    train_size = int(total * train_ratio)

    valid_size = int(total * valid_ratio)

    train_indices = indices[:train_size]

    valid_indices = indices[

        train_size:

        train_size + valid_size

    ]

    test_indices = indices[

        train_size + valid_size:

    ]

    train_dataset = [

        dataset[i]

        for i in train_indices

    ]

    valid_dataset = [

        dataset[i]

        for i in valid_indices

    ]

    test_dataset = [

        dataset[i]

        for i in test_indices

    ]

    return (

        train_dataset,

        valid_dataset,

        test_dataset

    )

# ============================================================

# 7. MODEL

# ============================================================

class GCN(torch.nn.Module):

    def __init__(self, num_features, num_classes):

        super().__init__()

        self.conv1 = GCNConv(

            in_channels=num_features,

            out_channels=32

        )

        self.conv2 = GCNConv(

            in_channels=32,

            out_channels=64

        )

        self.linear = torch.nn.Linear(

            64,

            num_classes

        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = self.conv2(x, edge_index)

        x = F.relu(x)

        x = self.linear(x)

        return x

# ============================================================

# 8. TRAIN FUNCTION

# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(

            batch.x,

            batch.edge_index

        )

        loss = criterion(

            out,

            batch.y

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss

# ============================================================

# 9. TEST FUNCTION - Node level

# ============================================================

def test_node_level(model, loader, device):

    model.eval()

    correct = 0

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            out = model(

                batch.x,

                batch.edge_index

            )

            pred = out.argmax(dim=1)

            correct += int(

                (pred == batch.y).sum()

            )

            all_preds.extend(

                pred.cpu().numpy()

            )

            all_labels.extend(

                batch.y.cpu().numpy()

            )

    acc = correct / len(all_labels)

    cm = confusion_matrix(

        all_labels,

        all_preds,

        labels=list(range(NUM_CLASSES))

    )

    return acc, cm

# ============================================================

# 10. TEST FUNCTION - Case level strict accuracy

# ============================================================

def test_case_level(model, loader, device):

    model.eval()

    total_cases = 0

    correct_cases = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            out = model(

                batch.x,

                batch.edge_index

            )

            pred = out.argmax(dim=1)

            ptr = batch.ptr

            for i in range(len(ptr) - 1):

                start = ptr[i]

                end = ptr[i + 1]

                case_pred = pred[start:end]

                case_true = batch.y[start:end]

                total_cases += 1

                if torch.equal(case_pred, case_true):

                    correct_cases += 1

    acc = correct_cases / total_cases

    return acc, correct_cases, total_cases

# ============================================================

# 11. MAIN

# ============================================================

def main():

    log_path = setup_experiment_log(

        experiment_name="Experiment"

    )

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------

    # Dataset

    # --------------------------------------------------------

    if VALIDATION_EXCEL_FILE is None:

        full_dataset = load_dataset_from_excel(EXCEL_FILE)

        (

            train_dataset,

            valid_dataset,

            test_dataset

        ) = split_dataset(

            full_dataset,

            TRAIN_RATIO,

            VALID_RATIO,

            TEST_RATIO

        )

    else:

        # Split the original training data into training and validation sets

        full_train_dataset = load_dataset_from_excel(EXCEL_FILE)

        train_dataset, valid_dataset, _ = split_dataset(

            full_train_dataset,

            TRAIN_RATIO / (TRAIN_RATIO + VALID_RATIO),

            VALID_RATIO / (TRAIN_RATIO + VALID_RATIO),

            0.0

        )

        # Use the separate Excel file as the final test dataset

        test_dataset = load_dataset_from_excel(

            VALIDATION_EXCEL_FILE

        )

    print()

    print("Train cases:", len(train_dataset))

    print("Validation cases:", len(valid_dataset))

    print("Test cases:", len(test_dataset))

    num_features = train_dataset[0].num_node_features

    print("Number of node features:", num_features)

    # --------------------------------------------------------

    # DataLoader

    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )

    # --------------------------------------------------------

    # Device / Model

    # --------------------------------------------------------

    device = torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"

    )

    print("Device:", device)

    model = GCN(

        num_features=num_features,

        num_classes=NUM_CLASSES

    ).to(device)

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )

    criterion = torch.nn.CrossEntropyLoss()

    #######################################################################

    MLP_model = MLPBaseline(

        input_dim=num_features,

        hidden_dim=64,

        num_classes=NUM_CLASSES,

        dropout=0.0

    ).to(device)

    MLP_optimizer = torch.optim.Adam(

        MLP_model.parameters(),

        lr=LEARNING_RATE

    )

    MLP_criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------

    # Train loop

    # --------------------------------------------------------

    last_cm = None

    for epoch in range(1, EPOCHS + 1):

        loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device

        )

        MLP_loss = train_one_epoch(

            MLP_model,

            train_loader,

            MLP_optimizer,

            MLP_criterion,

            device

        )

        acc, cm = test_node_level(

            model,

            valid_loader,

            device

        )

        MLP_acc, MLP_cm = test_node_level(

                    MLP_model,

                    valid_loader,

                    device

                )

        last_cm = cm

        MLP_last_cm = MLP_cm

        case_acc, correct_cases, total_cases = test_case_level(

            model,

            valid_loader,

            device

        )

        MLP_case_acc, MLP_correct_cases, MLP_total_cases = test_case_level(

            MLP_model,

            valid_loader,

            device

        )

        print(

            f"Epoch {epoch:04d} | "

            f"GCN Loss {loss:.4f} | "

            f"GCN Node Acc {acc:.4f} | GCN Geometry Acc {case_acc:.4f} | "

            f"MLP Loss {MLP_loss:.4f} | MLP Node Acc {MLP_acc:.4f} | MLP Geometry Acc {MLP_case_acc:.4f}"

        )

    # --------------------------------------------------------

    # Final result

    # --------------------------------------------------------

    print()

    print("Validation : Node-level Confusion Matrix")

    print(last_cm)

    case_acc, correct_cases, total_cases = test_case_level(

        model,

        test_loader,

        device

    )

    MLP_case_acc, MLP_correct_cases, MLP_total_cases = test_case_level(

        MLP_model,

        test_loader,

        device

    )

    print()

    print("Train cases      :", len(train_dataset))

    print("Validation cases :", len(valid_dataset))

    print("Test cases       :", len(test_dataset))

    print()

    print("GCN Case-level Strict Accuracy")

    print(

        f"{correct_cases} / {total_cases} = {case_acc:.4f}"

    )

    print()

    print("MLP Case-level Strict Accuracy")

    print(

        f"{MLP_correct_cases} / {MLP_total_cases} = {MLP_case_acc:.4f}"

    )

    test_acc, test_cm = test_node_level(model, test_loader, device)

    print("GCN Test Node Accuracy:", test_acc)

    print("GCN Test Node-level Confusion Matrix")

    print(test_cm)

    MLP_test_acc, MLP_test_cm = test_node_level(MLP_model, test_loader, device)

    print("MLP Test Node Accuracy:", MLP_test_acc)

    print("MLP Test Node-level Confusion Matrix")

    print(MLP_test_cm)

if __name__ == "__main__":

    main()
