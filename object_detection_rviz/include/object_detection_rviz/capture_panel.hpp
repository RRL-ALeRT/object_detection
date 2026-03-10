#ifndef CAPTURE_PANEL_HPP
#define CAPTURE_PANEL_HPP

#ifndef Q_MOC_RUN
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#endif

#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <mutex>

namespace object_detection_rviz
{

class CapturePanel : public rviz_common::Panel
{
Q_OBJECT
public:
  CapturePanel(QWidget* parent = nullptr);
  virtual ~CapturePanel();

  virtual void onInitialize();
  virtual void load(const rviz_common::Config& config);
  virtual void save(rviz_common::Config config) const;

protected Q_SLOTS:
  void captureImage();
  void updateTopic();
  void updateSaveDir();

protected:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  
  QLineEdit* topic_editor_;
  QLineEdit* save_dir_editor_;
  QPushButton* capture_button_;
  QLabel* status_label_;
  
  QString current_topic_;
  QString save_dir_;
  
  sensor_msgs::msg::Image::ConstSharedPtr last_img_msg_;
  std::mutex mutex_;
  
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
};

} // namespace object_detection_rviz

#endif // CAPTURE_PANEL_HPP