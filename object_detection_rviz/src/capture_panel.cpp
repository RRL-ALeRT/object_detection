#include "object_detection_rviz/capture_panel.hpp"
#include <rviz_common/display_context.hpp>
#include <QHBoxLayout>
#include <QLabel>
#include <QDebug>
#include <pluginlib/class_list_macros.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>

namespace fs = std::filesystem;

namespace object_detection_rviz
{

CapturePanel::CapturePanel(QWidget* parent)
  : rviz_common::Panel(parent), current_topic_(""), save_dir_("")
{
  QVBoxLayout* layout = new QVBoxLayout;
  
  QHBoxLayout* topic_layout = new QHBoxLayout;
  topic_layout->addWidget(new QLabel("Topic:"));
  topic_editor_ = new QLineEdit;
  topic_layout->addWidget(topic_editor_);
  layout->addLayout(topic_layout);

  QHBoxLayout* dir_layout = new QHBoxLayout;
  dir_layout->addWidget(new QLabel("Save Dir:"));
  save_dir_editor_ = new QLineEdit;
  dir_layout->addWidget(save_dir_editor_);
  layout->addLayout(dir_layout);

  capture_button_ = new QPushButton("Capture Image");
  layout->addWidget(capture_button_);

  status_label_ = new QLabel("Ready.");
  layout->addWidget(status_label_);

  setLayout(layout);

  connect(topic_editor_, SIGNAL(editingFinished()), this, SLOT(updateTopic()));
  connect(save_dir_editor_, SIGNAL(editingFinished()), this, SLOT(updateSaveDir()));
  connect(capture_button_, SIGNAL(clicked()), this, SLOT(captureImage()));

  // Default values
  topic_editor_->setText("/camera/color/image_raw");
  save_dir_editor_->setText("/home/oliver/captured_images");
}

CapturePanel::~CapturePanel()
{
}

void CapturePanel::onInitialize()
{
  auto node_weak = getDisplayContext()->getRosNodeAbstraction();
  auto node_locked = node_weak.lock();
  if (node_locked) {
      node_ = node_locked->get_raw_node();
  }
  
  updateTopic();
  updateSaveDir();
}

void CapturePanel::updateTopic()
{
  QString new_topic = topic_editor_->text();
  if (new_topic != current_topic_)
  {
    current_topic_ = new_topic;
    if (current_topic_.isEmpty())
    {
       sub_.reset();
       return;
    }
    
    std::string topic_name = current_topic_.toStdString();
    
    if (node_) {
        // Subscribe
        try {
            sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
              topic_name,
              rclcpp::SensorDataQoS(),
              std::bind(&CapturePanel::imageCallback, this, std::placeholders::_1));
            status_label_->setText(QString("Subscribed to %1").arg(current_topic_));
        } catch (...) {
            status_label_->setText(QString("Error subscribing to %1").arg(current_topic_));
        }
    }
  }
}

void CapturePanel::updateSaveDir()
{
    QString new_dir = save_dir_editor_->text();
    if (new_dir != save_dir_) {
        save_dir_ = new_dir;
        try {
            fs::create_directories(save_dir_.toStdString());
        } catch (const fs::filesystem_error& e) {
             status_label_->setText(QString("Error creating dir: %1").arg(e.what()));
        }
    }
}

void CapturePanel::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    last_img_msg_ = msg;
}

void CapturePanel::captureImage()
{
    sensor_msgs::msg::Image::ConstSharedPtr img;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        img = last_img_msg_;
    }

    if (!img)
    {
        status_label_->setText("No image received yet.");
        return;
    }

    // Convert and Save
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        
        // Generate timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        std::string timestamp = ss.str();
        
        fs::path dir(save_dir_.toStdString());
        // Ensure dir exists
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
        
        std::string filename = "capture_" + timestamp + ".jpg";
        fs::path full_path = dir / filename;
        
        if (cv::imwrite(full_path.string(), cv_ptr->image)) {
            status_label_->setText(QString("Saved: %1").arg(QString::fromStdString(filename)));
        } else {
             status_label_->setText("Failed to save image.");
        }
    } catch (cv_bridge::Exception& e) {
        status_label_->setText(QString("cv_bridge exception: %1").arg(e.what()));
    } catch (std::exception& e) {
        status_label_->setText(QString("Exception: %1").arg(e.what()));
    }
}

void CapturePanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("Topic", topic_editor_->text());
  config.mapSetValue("SaveDir", save_dir_editor_->text());
}

void CapturePanel::load(const rviz_common::Config& config)
{
  rviz_common::Panel::load(config);
  QString topic;
  if (config.mapGetString("Topic", &topic))
  {
    topic_editor_->setText(topic);
    updateTopic();
  }
  QString dir;
  if (config.mapGetString("SaveDir", &dir))
  {
    save_dir_editor_->setText(dir);
    updateSaveDir();
  }
}

} // namespace object_detection_rviz

PLUGINLIB_EXPORT_CLASS(object_detection_rviz::CapturePanel, rviz_common::Panel)