import 'package:flutter/scheduler.dart';

class DataModel {
  static int? id;
  static double? predictedQuality;
  static double? DO;
  static double? BOD;
  static double? totalCaliform;
  static double? fecalCaliform;

  // DataModel.fromJson(Map<String, dynamic> json) {
  //   state = json['state'];
  // }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = <String, dynamic>{};
    data['DO'] = DO;
    data['BOD'] = BOD;
    data['totalCaliform'] = totalCaliform;
    data['fecalCaliform'] = fecalCaliform;

    return data;
  }
}
