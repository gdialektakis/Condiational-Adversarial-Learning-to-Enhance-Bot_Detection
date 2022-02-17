import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from yellowbrick import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.style.palettes import LINE_COLOR
from imblearn.over_sampling import ADASYN


def draw_plots_for_Adasyn():
    # Test on Augmented Adasyn data
    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    synthetic_test_data = pickle.load(open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))
    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    classifier = RandomForestClassifier(n_jobs=2, random_state=1)

    for training_set in ["Original", "AC-GAN"]: #"ADASYN", "CGAN",
        # ROC macro average
        aha_title = f"Precision-Recall Curve for {training_set}"

        class CustomRoc(PrecisionRecallCurve):
            # def set_title(self, title=None):
            #     self.ax.set_title(aha_title)

            def draw(self):
                """
                Renders ROC-AUC plot.
                Called internally by score, possibly more than once

                Returns
                -------
                ax : the axis with the plotted figure
                """

                # Dictionary keys for ROCAUC
                MACRO = "macro"
                MICRO = "micro"

                # Target Type Constants
                BINARY = "binary"
                MULTICLASS = "multiclass"

                colors = self.class_colors_[0: len(self.classes_)]
                n_classes = len(colors)
                print(colors)

                # If it's a binary decision, plot the single ROC curve
                if self.target_type_ == BINARY and not self.per_class:
                    self.ax.plot(
                        self.fpr[BINARY],
                        self.tpr[BINARY],
                        color="teal",
                        label="ROC for binary decision, AUC = {:0.2f}".format(
                            self.roc_auc[BINARY]
                        ),
                    )

                # If per-class plotting is requested, plot ROC curves for each class
                if self.per_class:
                    for i, color in zip(range(n_classes), colors):
                        self.ax.plot(
                            self.fpr[i],
                            self.tpr[i],
                            color=color,
                            label="ROC of class {}, AUC = {:0.2f}".format(
                                self.classes_[i], self.roc_auc[i]
                            ),
                        )

                # If requested, plot the ROC curve for the micro average
                if training_set == "Original":
                    temp_color = 'red'
                elif training_set == "ADASYN":
                    temp_color = 'green'
                elif training_set == "CGAN":
                    temp_color = 'blue'
                if training_set == "AC-GAN":
                    temp_color = 'teal'
                else:
                    temp_color = 'purple'

                if self.micro:
                    self.ax.plot(
                        self.fpr[MICRO],
                        self.tpr[MICRO],
                        linestyle="--",
                        color=temp_color,
                        label="micro-average ROC curve {}, AUC = {:0.5f}".format(
                            training_set,
                            self.roc_auc["micro"],
                        ),
                    )
                if self.macro:
                    self.ax.plot(
                        self.fpr[MACRO],
                        self.tpr[MACRO],
                        linestyle="--",
                        color=temp_color,
                        label="macro-average ROC curve for {} Augmented Data, AUC = {:0.5f}".format(
                            training_set,
                            self.roc_auc["macro"],
                        ),
                    )

                # Plot the line of no discrimination to compare the curve to.
                self.ax.plot([0, 1], [0, 1], linestyle=":", c=LINE_COLOR)
                return self.ax

        scaler = MinMaxScaler()
        y_train = original_train_data['label']
        y_test = test_data['label']
        y_test = y_test.astype('int')

        # Drop unwanted columns
        X_train = original_train_data.drop(['label'], axis=1)
        X_test = test_data.drop(['label'], axis=1)

        if training_set == "Original":
            print('Original')
            X_train = scaler.fit_transform(X=X_train)
            X_test = scaler.transform(X=X_test)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            X_plot_train, X_plot_test, y_plot_train, y_plot_test = X_train, X_test, y_train, y_test

        elif training_set == "ADASYN":
            print('ADASYN')
            adasyn = ADASYN(random_state=42, n_jobs=-1)
            X_train = scaler.fit_transform(X=X_train)
            X_test = scaler.transform(X=X_test)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            X_plot_train, y_plot_train = adasyn.fit_resample(X_train, y_train)
            X_plot_test, y_plot_test = X_test, y_test

        elif training_set == "CGAN":
            print('CGAN')
            synthetic_data = pickle.load(
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))
            original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
            augmented_df = original_train_data.append(synthetic_data)
            augmented_df = augmented_df.sample(frac=1)

            y_train = augmented_df['label']
            # Drop unwanted columns
            X_train = augmented_df.drop(['label'], axis=1)
            X_train = scaler.fit_transform(X=X_train)
            X_test = scaler.transform(X=X_test)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            X_plot_train, y_plot_train = X_train, y_train
            X_plot_test, y_plot_test = X_test, y_test

        else:
            print('AC-GAN')

            synthetic_data = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))
            original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
            augmented_df = original_train_data.append(synthetic_data)
            augmented_df = augmented_df.sample(frac=1)

            y_train = augmented_df['label']
            # Drop unwanted columns
            X_train = augmented_df.drop(['label'], axis=1)

            X_train = scaler.fit_transform(X=X_train)
            X_test = scaler.transform(X=X_test)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')

            X_plot_train, y_plot_train = X_train, y_train
            X_plot_test, y_plot_test = X_test, y_test

        viz_roc = CustomRoc(classifier, micro=True, per_class=False)
        viz_roc.fit(X_plot_train, y_plot_train)  # Fit the training data to the viz_roc
        viz_roc.score(X_plot_test, y_plot_test)  # Evaluate the model on the test data

    viz_roc.show()  # Finalize and show the figure


draw_plots_for_Adasyn()