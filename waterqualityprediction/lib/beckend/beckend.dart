import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../constants/constants.dart';
import '../models/project_model.dart';

class Server {
  static const url = 'http://127.0.0.1:5000';

  static Future<void> fetchState() async {
    const currentURL = '$url/get_State_names';

    try {
      final response = await http.get(
        Uri.parse(currentURL),
      );
      if (response.statusCode == 200) {
        final jsonMapData = jsonDecode(response.body) as Map<String, dynamic>;
        kStates = jsonMapData['State'];
      } else {
        print("error in backend");
      }
    } catch (err) {
      print(err);
    }
  }

  static Future<void> estimateQuality() async {
    const _url = '$url/predict_Water_Quality';

    try {
      var data = json.encode({
        'DO': DataModel.DO,
        'BOD': DataModel.BOD,
        'totalCaliform': DataModel.totalCaliform,
        'fecalCaliform': DataModel.fecalCaliform,
      });
      print(data);
      final response = await http.post(Uri.parse(_url), body: data);

      if (response.statusCode == 200) {
        final jsonMapData = jsonDecode(response.body) as Map<String, dynamic>;
        print(response.body);
        print(jsonMapData['estimated_Quality']);

        DataModel.predictedQuality = jsonMapData['estimated_Quality'];
        print(DataModel.predictedQuality);
      } else {
        print("error");
      }
    } catch (err) {
      print(err);
    }
  }
}
