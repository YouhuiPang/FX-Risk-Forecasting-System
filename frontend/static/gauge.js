
function getGaugePointerColor(riskValue) {
  if (riskValue < 1/6) return "#4CAF50";       // 深绿
  else if (riskValue < 2/6) return "#66BB6A";
  else if (riskValue < 3/6) return "#FFEB3B";
  else if (riskValue < 4/6) return "#FF9800";
  else if (riskValue < 5/6) return "#F44336";
  else return "#D32F2F";
}

// 自定义插件，在 doughnut 图上绘制指针
const pointerPlugin = {
  id: 'pointerPlugin',
  afterDraw: function(chart, args, options) {
    const { ctx } = chart;
    // 获取第一个数据点的中心点信息
    const meta = chart.getDatasetMeta(0).data[0];
    const centerX = meta.x;
    const centerY = meta.y;
    const outerRadius = meta.outerRadius;
    const innerRadius = meta.innerRadius;
    const pointerLength = (outerRadius + innerRadius) / 2;

    // 计算指针角度
    const riskPercentage = options.riskValue; // 风险百分比（如 42）
    const fraction = riskPercentage / 100;      // 转为0~1之间
    const rotation = chart.options.rotation;    // 如 270°（以弧度表示）
    const circumference = chart.options.circumference;
    const pointerAngle = rotation + fraction * circumference;

    // 获取指针颜色
    const pointerColor = getGaugePointerColor(fraction);

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(pointerAngle);

    // 绘制指针直线
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(pointerLength, 0);
    ctx.lineWidth = 4;
    ctx.strokeStyle = pointerColor;
    ctx.stroke();

    // 绘制指针箭头（三角形）
    ctx.beginPath();
    ctx.moveTo(pointerLength, 0);
    ctx.lineTo(pointerLength - 10, -5);
    ctx.lineTo(pointerLength - 10, 5);
    ctx.closePath();
    ctx.fillStyle = pointerColor;
    ctx.fill();

    ctx.restore();
  }
};

function initGauge(risk) {
  // risk 为 0~1 的小数（例如 0.42 表示 42%）
  const ctx = document.getElementById('riskGauge').getContext('2d');
  const mainColor = getGaugePointerColor(risk);

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Risk', 'Remaining'],
      datasets: [{
        data: [risk * 100, 100 - risk * 100],
        backgroundColor: [
          mainColor,
          '#2b2e44'
        ],
        borderWidth: 0
      }]
    },
    options: {
      rotation: 270,       // 从 270° 开始
      circumference: 180,  // 半圆
      cutout: '70%',       // 中空比例
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      }
    },
  });
}
