 $(window).load(function(){  
             $(".loading").fadeOut()
            })  
			
/****/
/****/
$(document).ready(function(){
	var whei=$(window).width()
	$("html").css({fontSize:whei/20})
	$(window).resize(function(){
		var whei=$(window).width()
	 $("html").css({fontSize:whei/20})
});
	});

 
$(function () {

echarts_1()
echarts_2()
echarts_3()

    function echarts_1() {
        var myChart = echarts.init(document.getElementById('echart1'));
        option = {

            tooltip: {
                trigger: 'item'
            },

            series: [
                {
                    name: '丰枯遭遇',
                    type: 'pie',
                    radius: '75%',
                    data: [
                        { value: 0.64, name: '丰枯异步' ,itemStyle: {color:'#D95319'}},
                        { value: 0.15, name: '同枯' ,itemStyle: {color:'#0072BD'}},
                        { value: 0.21, name: '同丰' ,itemStyle: {color:'#77AC30'}},
                    ]
                    // emphasis: {
                    //   itemStyle: {
                    //     shadowBlur: 10,
                    //     shadowOffsetX: 0,
                    //     // shadowColor: 'rgba(0, 0, 0, 0.5)'
                    //   }
                    // }
                }
            ]
        };
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);

        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }

    function echarts_3() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('echart3'));

        // const colors = ['#5470C6', '#91CC75', '#EE6666'];
        option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    crossStyle: {
                        color: '#999'
                    }
                }
            },
            toolbox: {
                feature: {
                    dataView: { show: true, readOnly: false },
                    magicType: { show: true, type: ['line', 'bar'] },
                    restore: { show: true },
                    saveAsImage: { show: true }
                }
            },
            legend: {
                data: ['洪泽湖', '骆马湖'],
                textStyle: {
                    color: 'white', // 设置图例的字体颜色为蓝色
                },
            },
            xAxis: [
                {
                    type: 'category',
                    data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    axisPointer: {
                        type: 'shadow'
                    },
                    axisLabel: {
                        color: 'white', // 设置横坐标刻度字体颜色为蓝色
                    },
                }
            ],
            yAxis: [
                {
                    type: 'value',
                    name: '流量',
                    min: 0,
                    // max: 250,
                    // interval: 50,
                    axisLabel: {
                        formatter: '{value} m³/s'
                    },
                    splitLine: {
                        show:false
                    },
                    axisLabel: {
                        color: 'white', // 设置横坐标刻度字体颜色为蓝色
                    },
                    nameTextStyle: {
                        color: 'white', // 设置纵坐标名称的字体颜色为红色
                    },

                },
                {
                    type: 'value',
                    name: '流量',
                    min: 0,
                    // max: 25,
                    // interval: 5,
                    axisLabel: {
                        formatter: '{value}  m³/s'
                    },
                    splitLine: {
                        show:false
                    },
                    axisLabel: {
                        color: 'white', // 设置纵坐标刻度字体颜色为红色
                    },
                    nameTextStyle: {
                        color: 'white', // 设置纵坐标名称的字体颜色为红色
                    },
                }
            ],
            series: [
                {
                    name: '洪泽湖',
                    type: 'bar',
                    tooltip: {
                        valueFormatter: function (value) {
                            return value + '  m³/s';
                        }
                    },
                    data: [
                        235,290,469,494,681,803,2650,2230,1570,929,610,356
                    ],
                    yAxisIndex: 0,
                    itemStyle: {
                        color: '#4DBEEE', // 设置柱状图的颜色为绿色
                    },
                },
                {
                    name: '骆马湖',
                    type: 'bar',
                    tooltip: {
                        valueFormatter: function (value) {
                            return value + '  m³/s';
                        }
                    },
                    data: [
                        40.8,33.7,34.9,3.8,49.2,63.2,480,558,366,182,91.4,64.7
                    ],
                    yAxisIndex: 1,
                    itemStyle: {
                        color: '#D95319', // 设置柱状图的颜色为绿色
                    },
                }
            ]
        };


        // option = {
        //     color: ['#1aa1ff', '#31c17b', '#ff6535'],
        //     tooltip: {
        //         trigger: 'axis',
        //         axisPointer: {
        //             type: 'shadow'
        //         }
        //     },
        //     grid: {
        //         left: '10',
        //         top:'10%',
        //         right: '0%',
        //         bottom: '3%',
        //         containLabel: true
        //     },
        //     xAxis: {
        //         data: ['2014', '2015', '2016', '2017', '2018', '2019'],
        //         axisLine: {show:false,},
        //         axisLabel: {
        //             color: 'rgba(255,255,255,.6)',
        //             fontSize: 14
        //         }
        //     },
        //     yAxis: [
        //         {
        //             type: 'value',
        //             name: '降水量',
        //             min: 0,
        //             max: 250,
        //             position: 'right',
        //             axisLabel: {
        //                 formatter: '{value} ml',
        //                 color: 'rgba(255,255,255,.6)',
        //                 fontSize: 14
        //             },
        //             nameTextStyle: {
        //             color: 'rgba(255,255,255,.6)',
        //             fontSize: 14
        //         },
        //             splitLine: {
        //             lineStyle: {
        //                 color: "rgba(255,255,255,.1)",
        //                 type: "dotted"
        //             }
        //         },
        //         },
        //         {
        //             type: 'value',
        //             name: '温度',
        //             min: 0,
        //             max: 25,
        //             position: 'left',
        //             axisLabel: {
        //                 formatter: '{value} °C',
        //                 color: 'rgba(255,255,255,.6)',
        //                 fontSize: 14
        //             },
        //             nameTextStyle: {
        //             color: 'rgba(255,255,255,.6)',
        //             fontSize: 14
        //         },
        //             splitLine: {
        //             lineStyle: {
        //                 color: "rgba(255,255,255,.1)",
        //                 type: "dotted"
        //             }
        //         },
        //         }
        //     ],
        //     series: [{
        //         type: 'bar',
        //         barWidth: '30',
        //         itemStyle: {
        //             normal: {
        //                 barBorderRadius: 2,
        //                 color: function(params) {
        //                     var colorList = [
        //                         '#0074c2','#00b59d','#00be2b','#abd300','#f4e000',
        //                         '#ffab00','#ff7100','#f00c00','#c90061', '#c900c7','#C6E579','#F4E001','#F0805A','#26C0C0'
        //                     ];
        //
        //                     return colorList[params.dataIndex]
        //
        //                 },
        //
        //                 label: {
        //
        //                     show: true,
        //                     position: 'top',
        //                     formatter: '{c}',
        //                     color: 'rgba(255,255,255,.4)',
        //                     fontSize: 12
        //                 }
        //
        //             }
        //
        //         },
        //         data: [ 5, 12, 35, 100, 150, 235]
        //
        //     }]
        // };
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
    function echarts_2() {
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('echart2'));


        option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    crossStyle: {
                        color: '#999'
                    }
                }
            },
            toolbox: {
                feature: {
                    dataView: { show: true, readOnly: false },
                    magicType: { show: true, type: ['line', 'bar'] },
                    restore: { show: true },
                    saveAsImage: { show: true }
                }
            },
            legend: {
                data: ['需水片区一', '需水片区二',"需水片区三"],
                textStyle: {
                    color: 'white', // 设置图例的字体颜色为蓝色
                },
            },
            xAxis: [
                {
                    type: 'category',
                    data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    axisPointer: {
                        type: 'shadow'
                    },
                    axisLabel: {
                        color: 'white', // 设置横坐标刻度字体颜色为蓝色
                    },
                }
            ],
            yAxis: [
                {
                    type: 'value',
                    name: '需水量',
                    min: 0,
                    // max: 250,
                    // interval: 50,
                    axisLabel: {
                        formatter: '{value} 亿m³'
                    },
                    splitLine: {
                        show:false
                    },
                    axisLabel: {
                        color: 'white', // 设置横坐标刻度字体颜色为蓝色
                    },
                    nameTextStyle: {
                        color: 'white', // 设置纵坐标名称的字体颜色为红色
                    },

                },
                {
                    type: 'value',
                    name: '需水量',
                    min: 0,
                    // max: 25,
                    // interval: 5,
                    axisLabel: {
                        formatter: '{value}  亿m³'
                    },
                    splitLine: {
                        show:false
                    },
                    axisLabel: {
                        color: 'white', // 设置纵坐标刻度字体颜色为红色
                    },
                    nameTextStyle: {
                        color: 'white', // 设置纵坐标名称的字体颜色为红色
                    },
                }
            ],
            series: [
                {
                    name: '需水片区一',
                    type: 'bar',
                    tooltip: {
                        valueFormatter: function (value) {
                            return value + '  亿m³';
                        }
                    },
                    data: [
                        2.08,2.14,4.81,4.04,4.34,4.35,16.87,13.01,11.92,5.44,2.22,2.26
                    ],
                    yAxisIndex: 0,
                    itemStyle: {
                        color: '#4DBEEE', // 设置柱状图的颜色为绿色
                    },
                },
                {
                    name: '需水片区二',
                    type: 'bar',
                    tooltip: {
                        valueFormatter: function (value) {
                            return value + '  亿m³';
                        }
                    },
                    data: [
                       1.19,1.21,2.01,1.79,1.87,5.59,4.45,4.13,2.19,1.24,1.25,1.34
                       ],
                    yAxisIndex: 1,
                    itemStyle: {
                        color: '#D95319', // 设置柱状图的颜色为绿色
                    },
                },
                {
                    name: '需水片区三',
                    type: 'bar',
                    tooltip: {
                        valueFormatter: function (value) {
                            return value + '  亿m³';
                        }
                    },
                    data: [
                      3.62,3.74,8.79,7.33,7.92,31.57,24.29,22.24,9.99,3.89,3.96,4.53
                             ],
                    yAxisIndex: 0,
                    itemStyle: {
                        color: '#77AC30', // 设置柱状图的颜色为绿色
                    }
                }
            ]
        };
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
	
})












