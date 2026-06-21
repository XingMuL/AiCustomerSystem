// ECharts 按需引入（确保所有用到的图表和组件都被注册）
import * as echarts from 'echarts/core'
import { BarChart, LineChart, PieChart, GaugeChart, ScatterChart, GraphChart, RadarChart } from 'echarts/charts'
import {
  TitleComponent, TooltipComponent, LegendComponent, GridComponent,
  ToolboxComponent, DataZoomComponent, MarkLineComponent, RadarComponent,
  AxisPointerComponent
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([
  BarChart, LineChart, PieChart, GaugeChart, ScatterChart, GraphChart, RadarChart,
  TitleComponent, TooltipComponent, LegendComponent, GridComponent,
  ToolboxComponent, DataZoomComponent, MarkLineComponent, RadarComponent,
  AxisPointerComponent,
  CanvasRenderer
])

export default echarts
