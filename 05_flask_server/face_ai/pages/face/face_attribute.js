//index.js
//获取应用实例
var time = null;
var myCanvas = null;
var windowHeight, windowWidth;
var type = null;
Page({
  data: {
    device: true,
    camera: true,
    x1: 0,
    y1: 0,
    x2: 0,
    y2: 0,
  },
  onLoad() {
    this.setData({
      ctx: wx.createCameraContext(),
      device: this.data.device,
    })
    wx.getSystemInfo({
      success: function (res) {
        console.log(res);
        // 屏幕宽度、高度
        windowHeight = res.windowHeight;
        windowWidth = res.windowWidth;
        console.log('height=' + res.windowHeight);
        console.log('width=' + res.windowWidth);
      }
    })
  },
  open() {
    this.setData({
      camera: true,
      //login_res: "定位中..."
    })
    type = "takePhoto";
    let ctx = wx.createCameraContext(this)
    let that = this
    time = setInterval(function () {
      if (type == "takePhoto") {
        console.log("begin takephoto")
        ctx.takePhoto({
          quality: "normal",
          success: (res) => {
            console.log(res.tempImagePath)
            var tempImagePath = res.tempImagePath
            wx.uploadFile({
              url: 'http://127.0.0.1:90/face_attribute',
              filePath: tempImagePath,
              name: 'file',
              header: { "Content-type": "multipart/form-data" },
              success: function (res) {
                if (res.data == "error") {
                }
                else {
                  console.log(res.data);
                  var attri = res.data.split(",");
                  
                  if (attri[0] == 1) {
                    that.setData({
                      eye_flag: "Wear Glasses"
                    })
                  }else{
                    that.setData({
                      eye_flag: "No Glasses"
                    })
                  }

                  if (attri[1] == 1) {
                    that.setData({
                      young_flag: "Young"
                    })
                  } else {
                    that.setData({
                      young_flag: "Old"
                    })
                  }

                  if (attri[2] == 1) {
                    that.setData({
                      male_flag: "Man"
                    })
                  } else {
                    that.setData({
                      male_flag: "Woman"
                    })
                  }

                  if (attri[3] == 1) {
                    that.setData({
                      smiling_flag: "Smile"
                    })
                  } else {
                    that.setData({
                      smiling_flag: "Not Smile"
                    })
                  }

                }

              },
            })
          }
        })
      }
    }, 500)
  },
  // 关闭模拟的相机界面
  close() {
    console.log("关闭相机");
    type = "endPhoto"
    this.setData({
      camera: true,
      //login_res: "停止"
    })
  },
})
