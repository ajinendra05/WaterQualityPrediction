import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

import '../beckend/beckend.dart';
import '../constants/constants.dart';
import '../data/data.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models/project_model.dart';
import 'widget/dropdownButton.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List dropDownListData = [
    {"title": "BCA", "value": "1"},
    {"title": "MCA", "value": "2"},
    {"title": "B.Tech", "value": "3"},
    {"title": "M.Tech", "value": "4"},
    {"title": "BCA", "value": "1"},
    {"title": "MCA", "value": "2"},
    {"title": "B.Tech", "value": "3"},
    {"title": "M.Tech", "value": "4"},
    {"title": "BCA", "value": "1"},
    {"title": "MCA", "value": "2"},
    {"title": "B.Tech", "value": "3"},
    {"title": "M.Tech", "value": "4"},
    {"title": "BCA", "value": "1"},
    {"title": "MCA", "value": "2"},
    {"title": "B.Tech", "value": "3"},
    {"title": "M.Tech", "value": "4"},
    {"title": "BCA", "value": "1"},
    {"title": "MCA", "value": "2"},
    {"title": "B.Tech", "value": "3"},
    {"title": "M.Tech", "value": "4"},
    {"title": "BCA", "value": "1"},
    {"title": "MCA", "value": "2"},
    {"title": "B.Tech", "value": "3"},
    {"title": "M.Tech", "value": "4"},
  ];

  @override
  Widget build(BuildContext context) {
    var screenSize = MediaQuery.of(context).size;
    final _formKey = GlobalKey<FormState>();
    String? riverName;
    String? monthName;
    double? disolvedO2;
    double? BOD;
    double? totalCaliform;
    double? fecalColiform;
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: const [
            Icon(
              Icons.circle_sharp,
              size: 12,
              color: Colors.black,
            ),
            SizedBox(
              width: 10,
            ),
            Text(
              "Untitled UI",
              style: TextStyle(color: Colors.black),
            )
          ],
        ),
        // foregroundColor: Colors.black,
        backgroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: MediaQuery.of(context).size.width,
              height: 200,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topRight,
                  end: Alignment.bottomLeft,
                  colors: [
                    kGradient1,
                    kGradient2,
                  ],
                ),
              ),
              child: Align(
                alignment: Alignment.bottomCenter,
                child: Padding(
                  padding: const EdgeInsets.all(10.0),
                  child: CircleAvatar(
                    radius: 60,
                    backgroundImage: AssetImage(imagePath),
                  ),
                ),
              ),
            ),
            Center(
                child: Text(
              name,
              style: kTitleText,
            )),
            Center(
                child: Text(
              "@$username",
              style: kSubTitleText,
            )),
            const SizedBox(
              height: 10,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(
                  width: 10,
                ),
              ],
            ),
            const SizedBox(
              height: 10,
            ),
            Center(
              child: SizedBox(
                width: MediaQuery.of(context).size.width * 0.8,
                child: screenSize.width > 1200
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.start,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                            Form(
                                key: _formKey,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    TextFormField(
                                      validator: (value) {
                                        if (value!.isEmpty) {
                                          return 'please enter valid Name';
                                        } else {
                                          return null;
                                        }
                                      },
                                      onSaved: (newValue) {
                                        riverName = newValue;
                                      },
                                      keyboardType: TextInputType.name,
                                      decoration: const InputDecoration(
                                          labelText: 'River Name'),
                                    ),

                                    TextFormField(
                                        validator: (value) {
                                          if (value!.isEmpty ||
                                              int.parse(value) > 12 ||
                                              int.parse(value) < 0) {
                                            return 'please enter valid Month';
                                          } else {
                                            return null;
                                          }
                                        },
                                        onSaved: (newValue) {
                                          monthName = newValue;
                                        },
                                        decoration: const InputDecoration(
                                            labelText: 'Month Number')),
                                    TextFormField(
                                        validator: (value) {
                                          if (value!.isEmpty) {
                                            return 'please enter valid value';
                                          } else {
                                            return null;
                                          }
                                        },
                                        onSaved: (newValue) {
                                          DataModel.DO =
                                              double.parse(newValue!);
                                        },
                                        decoration: const InputDecoration(
                                            labelText:
                                                'Dissolved oxygen (DO)')),
                                    TextFormField(
                                        validator: (value) {
                                          if (value!.isEmpty) {
                                            return 'please enter valid value';
                                          } else {
                                            return null;
                                          }
                                        },
                                        onSaved: (newValue) {
                                          DataModel.BOD =
                                              double.parse(newValue!);
                                        },
                                        decoration: const InputDecoration(
                                            labelText:
                                                'Biological oxygen demand (BOD)')),
                                    TextFormField(
                                        validator: (value) {
                                          if (value!.isEmpty) {
                                            return 'please enter valid value';
                                          } else {
                                            return null;
                                          }
                                        },
                                        onSaved: (newValue) {
                                          DataModel.totalCaliform =
                                              double.parse(newValue!);
                                        },
                                        decoration: const InputDecoration(
                                            labelText: 'Total Coliform')),
                                    TextFormField(
                                        validator: (value) {
                                          if (value!.isEmpty) {
                                            return 'please enter valid value';
                                          } else {
                                            return null;
                                          }
                                        },
                                        onSaved: (newValue) {
                                          DataModel.fecalCaliform =
                                              double.parse(newValue!);
                                        },
                                        decoration: const InputDecoration(
                                            labelText: 'Fecal Coliform')),

                                    const SizedBox(
                                      height: 10,
                                    ),
                                    // if (widget.isloading) const CircularProgressIndicator(),
                                    // if (!(widget.isloading))
                                    // ElevatedButton(
                                    //     onPressed: () {

                                    //     },
                                    //     child: Text('Predict Quality')),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                          elevation: 0,
                                          backgroundColor:
                                              Colors.purple.shade900,
                                          shape: RoundedRectangleBorder(
                                              borderRadius:
                                                  BorderRadius.circular(8))),
                                      onPressed: () async {
                                        final isValid =
                                            _formKey.currentState!.validate();
                                        _formKey.currentState!.save();
                                        await Server.estimateQuality();
                                        cancelAlertDialog(riverName);
                                      },
                                      child: Text('PREDICT PRICE',
                                          style: kTitleText.copyWith(
                                              fontSize: 24, color: kGradient2)),
                                    ),
                                  ],
                                ))
                          ])
                    : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Form(
                              key: _formKey,
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  TextFormField(
                                    validator: (value) {
                                      if (value!.isEmpty) {
                                        return 'please enter valid Name';
                                      } else {
                                        return null;
                                      }
                                    },
                                    onSaved: (newValue) {
                                      riverName = newValue;
                                    },
                                    keyboardType: TextInputType.name,
                                    decoration: const InputDecoration(
                                        labelText: 'River Name'),
                                  ),

                                  TextFormField(
                                      validator: (value) {
                                        if (value!.isEmpty ||
                                            int.parse(value) > 12 ||
                                            int.parse(value) < 0) {
                                          return 'please enter valid Month';
                                        } else {
                                          return null;
                                        }
                                      },
                                      onSaved: (newValue) {
                                        monthName = newValue;
                                      },
                                      decoration: const InputDecoration(
                                          labelText: 'Month Number')),
                                  TextFormField(
                                      validator: (value) {
                                        if (value!.isEmpty) {
                                          return 'please enter valid value';
                                        } else {
                                          return null;
                                        }
                                      },
                                      onSaved: (newValue) {
                                        DataModel.DO = double.parse(newValue!);
                                      },
                                      decoration: const InputDecoration(
                                          labelText: 'Dissolved oxygen (DO)')),
                                  TextFormField(
                                      validator: (value) {
                                        if (value!.isEmpty) {
                                          return 'please enter valid value';
                                        } else {
                                          return null;
                                        }
                                      },
                                      onSaved: (newValue) {
                                        DataModel.BOD = double.parse(newValue!);
                                      },
                                      decoration: const InputDecoration(
                                          labelText:
                                              'Biological oxygen demand (BOD)')),
                                  TextFormField(
                                      validator: (value) {
                                        if (value!.isEmpty) {
                                          return 'please enter valid value';
                                        } else {
                                          return null;
                                        }
                                      },
                                      onSaved: (newValue) {
                                        DataModel.totalCaliform =
                                            double.parse(newValue!);
                                      },
                                      decoration: const InputDecoration(
                                          labelText: 'Total Coliform')),
                                  TextFormField(
                                      validator: (value) {
                                        if (value!.isEmpty) {
                                          return 'please enter valid value';
                                        } else {
                                          return null;
                                        }
                                      },
                                      onSaved: (newValue) {
                                        DataModel.predictedQuality =
                                            double.parse(newValue!);
                                      },
                                      decoration: const InputDecoration(
                                          labelText: 'Fecal Coliform')),

                                  const SizedBox(
                                    height: 10,
                                  ),
                                  // if (widget.isloading) const CircularProgressIndicator(),
                                  // if (!(widget.isloading))
                                  // ElevatedButton(
                                  //     onPressed: () {

                                  //     },
                                  //     child: Text('Predict Quality')),
                                  ElevatedButton(
                                    style: ElevatedButton.styleFrom(
                                        elevation: 0,
                                        backgroundColor: Colors.purple.shade900,
                                        shape: RoundedRectangleBorder(
                                            borderRadius:
                                                BorderRadius.circular(8))),
                                    onPressed: () async {
                                      final isValid =
                                          _formKey.currentState!.validate();
                                      _formKey.currentState!.save();
                                      await Server.estimateQuality();
                                      cancelAlertDialog(riverName);
                                    },
                                    child: Text('PREDICT PRICE',
                                        style: kTitleText.copyWith(
                                            fontSize: 24, color: kGradient2)),
                                  ),
                                ],
                              ))
                        ],
                      ),
              ),
            ),
            Center(
              child: SizedBox(
                  width: MediaQuery.of(context).size.width * 0.8,
                  child: screenSize.width > 1000 ? null : null),
            ),
          ],
        ),
      ),
    );
  }

  void cancelAlertDialog(String? riverName) {
    var height = MediaQuery.of(context).size.height;
    var width = MediaQuery.of(context).size.width;
    showDialog(
      context: context,
      builder: (context) => Container(
        height: height * 0.06,
        width: width,
        child: SimpleDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          alignment: Alignment.center,
          title: Stack(alignment: Alignment.center, children: [
            Icon(
              Icons.check,
              size: height * 0.05,
              color: Colors.green,
            ),
            Icon(
              Icons.circle_outlined,
              size: height * 0.07,
              color: Colors.green,
            ),
          ]),
          contentPadding: EdgeInsets.symmetric(horizontal: width * 0.04),
          children: [
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  height: height * 0.02,
                ),
                Text(
                  'Predicted Water Quality of ${riverName} in given situation',
                  style: kSubTitleText.copyWith(
                      fontSize: 25, fontWeight: FontWeight.w700),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text('${DataModel.predictedQuality}',
                    style: kTitleText.copyWith(
                        fontSize: 30, fontWeight: FontWeight.bold)),
                SizedBox(
                  height: height * 0.05,
                ),
                SizedBox(
                  height: height * 0.02,
                ),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                      elevation: 0,
                      backgroundColor: Colors.grey.shade400,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8))),
                  onPressed: () {},
                  child: const Text('Am I correct :)',
                      style: TextStyle(
                        color: Colors.black26,
                        fontFamily: 'Manrope',
                        fontSize: 15.5,
                        fontWeight: FontWeight.w900,
                      )),
                ),
                SizedBox(
                  height: height * 0.04,
                )
              ],
            )
          ],
        ),
      ),
    );
  }
}
