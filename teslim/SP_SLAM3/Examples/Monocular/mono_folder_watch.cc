/**
 * mono_folder_watch.cc — TEKNOFEST Görev 2 headless SLAM sürücüsü (FAZ 8.1)
 *
 * inbox/ klasörünü izler; <frame_id>.png/.jpg/.jpeg/.webp geldikçe SIRAYLA işler,
 * her kare için kamera→dünya pozunu (Twc = Tcw.inverse()) outbox/pose.txt'e
 * flush'lı tek satır olarak ekler:
 *     frame_id tx ty tz qx qy qz qw state
 * state ∈ {NOT_READY, NO_IMAGE, NOT_INITIALIZED, OK, RECENTLY_LOST, LOST, UNKNOWN}
 *
 * Durdurma: inbox/STOP adlı bir dosya oluşturulunca temiz kapanır.
 * Kullanım:
 *   ./mono_folder_watch vocab settings inbox_dir outbox_dir
 * Not: superpoint.pt göreli yüklendiği için SP_SLAM3 kökünden çalıştırın.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cctype>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/stat.h>
#include <dirent.h>

#include <Eigen/Dense>
#include <System.h>

using namespace std;

static const char* StateName(int s)
{
    switch (s) {
        case -1: return "NOT_READY";
        case 0:  return "NO_IMAGE";
        case 1:  return "NOT_INITIALIZED";
        case 2:  return "OK";
        case 3:  return "RECENTLY_LOST";
        case 4:  return "LOST";
        default: return "UNKNOWN";
    }
}

// inbox'taki işlenmemiş <sayı>.<uzantı> dosyalarını artan frame_id sırasıyla döndür
static vector<pair<long,string>> ScanInbox(const string &inbox,
                                           const set<long> &processed)
{
    vector<pair<long,string>> out;
    DIR *dir = opendir(inbox.c_str());
    if (!dir) return out;
    struct dirent *e;
    while ((e = readdir(dir)) != nullptr) {
        string name = e->d_name;
        size_t dot = name.find_last_of('.');
        if (dot == string::npos || dot == 0) continue;
        string stem = name.substr(0, dot);
        string ext  = name.substr(dot + 1);
        for (char &c : ext) c = (char)tolower(c);
        if (ext != "png" && ext != "jpg" && ext != "jpeg" &&
            ext != "webp" && ext != "bmp") continue;
        if (!all_of(stem.begin(), stem.end(), ::isdigit)) continue;
        long id = stol(stem);
        if (processed.count(id)) continue;
        out.push_back({id, inbox + "/" + name});
    }
    closedir(dir);
    sort(out.begin(), out.end());
    return out;
}

static bool FileExists(const string &p)
{
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        cerr << "Kullanim: ./mono_folder_watch vocab settings inbox_dir outbox_dir [viewer]" << endl;
        return 1;
    }
    const bool use_viewer = (argc == 6 && string(argv[5]) == "viewer");
    const string vocab_path    = argv[1];
    const string settings_path = argv[2];
    const string inbox         = argv[3];
    const string outbox        = argv[4];

    // Kalibrasyon çözünürlüğü + fps'i ayar dosyasından oku
    int camW = 0, camH = 0;
    double fps = 7.5;
    {
        cv::FileStorage fs(settings_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            cerr << "Ayar dosyasi acilamadi: " << settings_path << endl;
            return 1;
        }
        if (!fs["Camera.width"].empty())  camW = (int)fs["Camera.width"];
        if (!fs["Camera.height"].empty()) camH = (int)fs["Camera.height"];
        if (!fs["Camera.fps"].empty())    fps  = (double)fs["Camera.fps"];
    }
    if (fps <= 0) fps = 7.5;

    mkdir(outbox.c_str(), 0775);
    const string pose_path = outbox + "/pose.txt";
    // Taze oturum: pose.txt'i sıfırla
    ofstream pose(pose_path, ios::trunc);
    if (!pose.is_open()) {
        cerr << "pose.txt yazilamiyor: " << pose_path << endl;
        return 1;
    }
    pose << fixed << setprecision(9);

    // Görselleştirici: yarışmada KAPALI; hata ayıklama/demoda "viewer" argümanıyla açık
    ORB_SLAM3::System SLAM(vocab_path, settings_path,
                           ORB_SLAM3::System::MONOCULAR, use_viewer);

    cout << "[folder_watch] hazir. inbox=" << inbox << " outbox=" << outbox
         << " fps=" << fps << " kalibrasyon=" << camW << "x" << camH << endl;
    // Köprünün "sistem hazır" bilmesi için işaret dosyası
    { ofstream ready(outbox + "/READY"); ready << "1\n"; }

    set<long> processed;
    const string stop_file = inbox + "/STOP";

    while (true) {
        if (FileExists(stop_file)) {
            cout << "[folder_watch] STOP gorüldu, kapaniliyor..." << endl;
            break;
        }

        vector<pair<long,string>> todo = ScanInbox(inbox, processed);
        if (todo.empty()) {
            this_thread::sleep_for(chrono::milliseconds(5));
            continue;
        }

        for (const auto &item : todo) {
            const long id = item.first;
            const string &path = item.second;

            cv::Mat im = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (im.empty()) {
                // Yazımı henüz bitmemiş olabilir; bir kez kısa bekleyip tekrar dene
                this_thread::sleep_for(chrono::milliseconds(20));
                im = cv::imread(path, cv::IMREAD_UNCHANGED);
                if (im.empty()) {
                    cerr << "[folder_watch] okunamadi, atlaniyor: " << path << endl;
                    processed.insert(id);
                    pose << id << " 0 0 0 0 0 0 1 READ_FAIL" << endl;
                    continue;
                }
            }

            // Kalibrasyon çözünürlüğüne küçült (köprü zaten küçültür; sigorta)
            if (camW > 0 && camH > 0 && (im.cols != camW || im.rows != camH))
                cv::resize(im, im, cv::Size(camW, camH), 0, 0, cv::INTER_AREA);

            const double t = (double)id / fps;
            cv::Mat Tcw = SLAM.TrackMonocular(im, t);   // 4x4 float, bos ise izleme yok
            const int state = SLAM.GetTrackingState();

            if (Tcw.empty() || Tcw.rows != 4 || Tcw.cols != 4) {
                pose << id << " 0 0 0 0 0 0 1 " << StateName(state) << endl;
            } else {
                // Twc = Tcw^-1 : Rwc = R^T, twc = -R^T * tcw
                cv::Mat R = Tcw.rowRange(0,3).colRange(0,3);
                cv::Mat tc = Tcw.rowRange(0,3).col(3);
                cv::Mat Rwc = R.t();
                cv::Mat twc = -Rwc * tc;

                Eigen::Matrix3f Re;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        Re(r,c) = Rwc.at<float>(r,c);
                Eigen::Quaternionf q(Re);
                q.normalize();

                pose << id << " "
                     << twc.at<float>(0) << " " << twc.at<float>(1) << " "
                     << twc.at<float>(2) << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                     << StateName(state) << endl;   // endl = '\n' + flush
            }

            processed.insert(id);

            if (FileExists(stop_file)) break;
        }
    }

    SLAM.Shutdown();
    pose.close();
    cout << "[folder_watch] kapandi. islenen kare: " << processed.size() << endl;
    return 0;
}
