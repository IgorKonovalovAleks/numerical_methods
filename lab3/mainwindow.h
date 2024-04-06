#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#pragma once
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QButtonGroup>
#include <QLabel>
#include <QSlider>
#include <QStringList>
#include <QTableWidget>
#include <QHeaderView>
#include <QMessageBox>
#include <QtDataVisualization>
#include <QStyleFactory>
#include <QThread>
#include <vector>
#include "solver.h"

#define type_d high_precision_type

using high_precision_type = boost::multiprecision::cpp_dec_float_50;

class ProgressWindow : public QWidget {
    Q_OBJECT
public:
    ProgressWindow(solver *slvr, QWidget *parent = nullptr) : QWidget(parent), slvr(slvr){
        progwin = new QDialog(this);
        progwin->setWindowTitle("Выполнение...");
        progressBar = new QProgressBar(this);
        progressBar->setRange(0, 100);
        accuracyLabel = new QLabel(this);
        accuracyLabel->setText("Я делал этот прогресс бар 4 часа");

        QVBoxLayout *layout = new QVBoxLayout(this);
        layout->addWidget(progressBar);
        layout->addWidget(accuracyLabel);
        progwin->setLayout(layout);
        connect(slvr, &solver::progressUpdate, this, &ProgressWindow::handleProgressUpdate);
        progwin->show();
    }
    void closeWindow(){
        progwin->close();
    }
public slots:
    void handleProgressUpdate(int progress, type_d accuracy, qint64 elapsed, int curr_it) {
        progressBar->setValue(progress);
        progwin->setWindowTitle(QString("Выполнение... %1%").arg(progress));
        accuracyLabel->setText(QString("Текущая точность: " + QString::number(static_cast<double>(accuracy)) +
                                       "\nТекущий номер итерации: " + QString::number(curr_it) +
                                       "\nВремя выполнения программы: " + QString::number(elapsed / 1000) + " секунд\n" +
                                       "Среднее время итерации: " +  QString::number(static_cast<float>(elapsed) / (1000 * curr_it)) + " сек/итер"));
    }

private:
    QDialog *progwin;
    solver *slvr;
    QProgressBar *progressBar;
    QLabel *accuracyLabel;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    Q3DSurface *view;
    QLineEdit *xn;
    QLineEdit *yn;
    QLineEdit *eps;
    QLineEdit *maxsteps;
    QLineEdit *w;
    QLineEdit *eps2;
    QLineEdit *maxsteps2;
    QLineEdit *w2;
    QLineEdit *intervals;
    QLabel* summary;
    QSlider *slider;
    QLabel *valueLabel;
    QSurface3DSeries *iterations[10];
    QSurface3DSeries *iterations2[10];
    QSurface3DSeries *dataSeries = nullptr;
    QSurface3DSeries *dataPSeries = nullptr;
    QSurface3DSeries *dataTrueSeries = nullptr;
    ProgressWindow *prog = nullptr;
    solver slv;
    solver slv2;
    std::vector<std::vector<std::vector<type_d>>> v;
    std::vector<std::vector<std::vector<type_d>>> v2;
    std::vector<std::vector<std::vector<type_d>>> z;
    type_d a, b, c, d;
    int selectedTask = Functions::test;
    int selectedMeth = 0;
    bool activePlot{false};
    bool lastPlotButton{false};
    bool help = 0;
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

public slots:
    void showGraph();
    Q_INVOKABLE void handleSolveFinished();
    void drawGraph();
    void showTrueGraph();
    void showPGraph();
    void removeGraph();
    void removeTrueGraph();
    void removePGraph();
    void showTable();
    void showSummary();
    void setT(int t);
    void onRadioButtonClicked();
    void solveInBackground(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps, int m_it, std::vector<std::vector<std::vector<type_d>>>& v, std::vector<std::vector<std::vector<type_d>>>& z);
    void solveInBackground(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps, int m_it, std::vector<std::vector<std::vector<type_d>>>& v, type_d eps2, int m_it2, std::vector<std::vector<std::vector<type_d>>>& v2);
signals:
    void solverFinished2();
    void solverFinished3();
};

#endif // MAINWINDOW_H
