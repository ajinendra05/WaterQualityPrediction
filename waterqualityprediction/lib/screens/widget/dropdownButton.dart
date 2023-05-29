// import 'package:dropdown_button2/dropdown_button2.dart';
// import 'package:flutter/material.dart';

// import '../../models/project_model.dart';

// class MyDropDownButton extends StatefulWidget {
//   List dropDownListData;
//   int id;

//   MyDropDownButton(
//       {super.key, required this.dropDownListData, required this.id});

//   @override
//   State<MyDropDownButton> createState() => _DropDownButtonState();
// }

// class _DropDownButtonState extends State<MyDropDownButton> {
//   // List dropDownListData = widget.dropDownListData;

//   String defaultValue = "";

//   String secondDropDown = "";

//   @override
//   Widget build(BuildContext context) {
//     return InputDecorator(
//       decoration: InputDecoration(
//         border: OutlineInputBorder(borderRadius: BorderRadius.circular(15.0)),
//         contentPadding: const EdgeInsets.all(10),
//       ),
//       child: DropdownButtonHideUnderline(
//         child: DropdownButton<String>(
//             isDense: true,
//             value: defaultValue,
//             isExpanded: true,
//             menuMaxHeight: 350,
//             items: [
//               const DropdownMenuItem(
//                 value: "",
//                 child: Text(
//                   "Select Here",
//                 ),
//               ),
//               ...widget.dropDownListData.map<DropdownMenuItem<String>>((data) {
//                 return DropdownMenuItem(
//                   value: data,
//                   child: Text(data),
//                 );
//               }).toList(),
//             ],
//             onChanged: (value) {
//               switch (widget.id) {
//                 case 1:
//                   DataModel.state = value;
//                   break;
//                 case 2:
//                   DataModel.district = value;
//                   break;
//                 case 3:
//                   DataModel.variety = value;
//                   break;
//                 case 4:
//                   DataModel.month = value;
//                   break;
//                 case 5:
//                   if (value == 'True') {
//                     DataModel.isProducing = 1;
//                   } else {
//                     DataModel.isProducing = 0;
//                   }
//                   break;
//                 case 6:
//                   if (value == 'True') {
//                     DataModel.isHarvesting = 1;
//                   } else {
//                     DataModel.isHarvesting = 0;
//                   }
//                   break;
//                 case 7:
//                   if (value == 'True') {
//                     DataModel.isDrought = 1;
//                   } else {
//                     DataModel.isDrought = 0;
//                   }
//                   break;
//               }
//               setState(() {
//                 defaultValue = value!;
//               });
//             }),
//       ),
//     );
//   }
// }
