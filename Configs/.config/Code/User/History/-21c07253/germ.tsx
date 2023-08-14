import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, FlatList, StyleSheet, Alert } from 'react-native';
import AntDesign from 'react-native-vector-icons/AntDesign';
import { MultiSelect } from 'react-native-element-dropdown';

import AsyncStorage from '@react-native-async-storage/async-storage';
import { getQuestionLabels } from '../../utils/questionloader';

const StartScreen = ({ navigation }) => {
  const [name, setName] = useState('');
  const [names, setNames] = useState([]);
  const [questionsSets, setQuestionsSets] = useState([]);
  const [selectedItems, setSelectedItems] = useState([]);
  const [pointsToWin, setPointsToWin] = useState(10);
  const [pointsToWinDisplay, setPointsToWinDisplay] = useState('10');

  const renderDataItem = (item) => {

    return (
      <View style={styles.item}>
        <Text style={styles.selectedTextStyle}>{item.label}</Text>
      </View>

    );

  };
  //save selected items
  const saveSelectedItems = async (item) => {
    try {
      await AsyncStorage.setItem('customSet', JSON.stringify(item));
    } catch (error) {
      console.log('Error saving custom set:', error);
    }
  };


  useEffect(() => {
    // Load the saved names when the component mounts

    const loadNames = async () => {
      try {
        const savedNames = await AsyncStorage.getItem('names');
        if (savedNames !== null) {
          setNames(JSON.parse(savedNames));
        }
      } catch (error) {
        console.log('Error loading names:', error);
      }
    };

    loadNames();


    const loadSelectedSets = async () => {
      try {
        const savedSelectedSets = await AsyncStorage.getItem('customSet');
        if (savedSelectedSets !== null) {

          setSelectedItems(JSON.parse(savedSelectedSets));
        }
      } catch (error) {
        console.log('Error loading selected sets:', error);
      }
    };

    loadSelectedSets();

    const loadPointsToWin = async () => {
      try {
        const savedPointsToWin = await AsyncStorage.getItem('pointsToWin');
        if (savedPointsToWin !== null) {
          setPointsToWin(parseInt(savedPointsToWin));
          setPointsToWinDisplay(savedPointsToWin);
        }
      } catch (error) {
        console.log('Error loading points to win:', error);
      }
    }
    loadPointsToWin();

  }, []);

  // Load the question set specified in route.params.id. First use the sets from constants/questions.ts, then use the custom sets from AsyncStorage
  useEffect(() => {
    const getCustomSet = async () => {

      setQuestionsSets(await getQuestionLabels());
    }

    // copy the initial set from constants/questions.ts
    const unsubscribe = navigation.addListener('focus', () => {
      getCustomSet();
    });
    getCustomSet();
  }, []);


  useEffect(() => {
    // set navigation header button
    navigation.setOptions({
      headerRight: () => (
        // View with width of 50 to make the button easier to press
        <TouchableOpacity
          style={{ width: 80, height: 40, flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}
          onPress={() => navigation.navigate('Eigene Sets')}
        >
          <AntDesign name="edit" size={24} color="black" />
        </TouchableOpacity>
      ),
    });

  }, [navigation]);

  useEffect(() => {
    // Save the names whenever the names state changes
    saveNames();
  }, [names]);

  useEffect(() => {
    // Save the points to win whenever the pointsToWin state changes
    savePointsToWin();
  }
    , [pointsToWin]);

  const setSelectedItemsHelper = (item) => {
    setSelectedItems(item);
    saveSelectedItems(item);
  }


  const saveNames = async () => {
    try {
      const namesToSave = JSON.stringify(names);
      await AsyncStorage.setItem('names', namesToSave);
    } catch (error) {
      console.log('Error saving names:', error);
    }
  };

  const savePointsToWin = async () => {
    try {
      await AsyncStorage.setItem('pointsToWin', pointsToWin.toString());
    } catch (error) {
      console.log('Error saving points to win:', error);
    }
  };

  const handleAddName = () => {
    if (name.trim() !== '') {
      if (names.length < 4) {
        setNames([...names, name]);
        setName('');
      } else {
        Alert.alert('Zu viele Spieler', 'Maximal 4 Spieler möglich!');
      }

    }
  };

  const handlePointsToWinText = (points: string) => {
   if (points.trim() == '') {
    setPointsToWinDisplay('');
   } else {
    // remove chars that are not numbers
    const pointsToWinDisplay = points.replace(/[^0-9]/g, '');
    setPointsToWinDisplay(pointsToWinDisplay);
    setPointsToWin(parseInt(pointsToWinDisplay));
    }
  };


  const handleStartButton = () => {
    if (names.length == 0) {
      Alert.alert('Keine Spieler', 'Bitte mindestens einen Namen eingeben!');
    } else if (selectedItems.length == 0) {
      Alert.alert("Kein Set ausgewählt", "Bitte wähle ein Set aus!")
    } else {
      var chosenPointsToWin = pointsToWin;
      if (pointsToWinDisplay == '' || pointsToWinDisplay == '0') {
        chosenPointsToWin = 10;
        setPointsToWin(10);
      }
      navigation.navigate('RateKunst', { names: names, setID: selectedItems, pointsToWin: chosenPointsToWin });
    }
  }

  const handleRemoveName = (index: number) => {
    const updatedNames = [...names];
    updatedNames.splice(index, 1);
    setNames(updatedNames);
  };

  return (

    <View style={[styles.container]}>
      { /* Picker for question set */}
      <View style={styles.pickercontainer}>
        <MultiSelect
          style={styles.dropdown}
          placeholderStyle={styles.placeholderStyle}
          selectedTextStyle={styles.selectedTextStyle}
          inputSearchStyle={styles.inputSearchStyle}
          activeColor="tomato"
          iconStyle={styles.iconStyle}
          data={questionsSets}
          labelField="label"
          valueField="value"
          placeholder="Themensets auswählen"
          value={selectedItems}
          search
          searchPlaceholder="Suchen..."
          onChange={item => {
            setSelectedItemsHelper(item);
          }}
          renderLeftIcon={() => (
            <AntDesign
              style={styles.icon}
              color="black"
              name="folderopen"
              size={20}
            />
          )}

          renderItem={renderDataItem}
          renderSelectedItem={(item, unSelect) => (
            <TouchableOpacity onPress={() => unSelect && unSelect(item)}>
              <View style={styles.selectedStyle}>
                <Text style={styles.textSelectedStyle}>{item.label}</Text>
                <AntDesign color="black" name="delete" size={17} />
              </View>
            </TouchableOpacity>
          )}

        />
      </View>
      { /* List of names */}
      <Text style={styles.title}>Namen</Text>
      <FlatList
        data={names}
        renderItem={({ item, index }) => (
          <View style={styles.nameContainer}>
            <Text style={styles.names}>{item}</Text>
            <TouchableOpacity onPress={() => handleRemoveName(index)}>
              <AntDesign name="delete" size={24} color="white" />
            </TouchableOpacity>
          </View>
        )}
        keyExtractor={(item, index) => index.toString()}
      />
      <View style={styles.inputContainerNumber}>
        <Text style={styles.textPointsToWin}>Punkte um zu gewinnen:</Text>
        <TextInput
          style={styles.inputNumber}
          placeholder="10"
          placeholderTextColor={'#a9a9a9'}
          value={pointsToWinDisplay}
          onChangeText={(text) => {handlePointsToWinText(text)}}
          keyboardType="numeric"
        />

      </View>
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          placeholder="Name eingeben"
          placeholderTextColor={'#a9a9a9'}
          value={name}
          onChangeText={(text) => setName(text)}
        />
        <TouchableOpacity style={styles.addButton} onPress={handleAddName}>
          <AntDesign name="plussquareo" size={24} color="black" />
        </TouchableOpacity>
      </View>

      {/* Button to start the game */}
      <TouchableOpacity style={styles.startButton} onPress={() => handleStartButton()}>
        <Text style={styles.startButtonText}>Start</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    // dark gray background
    backgroundColor: '#1f1f23'
  },
  textPointsToWin: {
    fontSize: 18,
    marginBottom: 32,
    color: 'white',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
    color: 'white',

  },
  names: {
    fontSize: 18,
    marginBottom: 8,
    color: 'white',
  },
  nameContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  inputContainerNumber: {
    flexDirection: 'row',
    alignItems: 'center',
    // algin everything in the middle
    justifyContent: 'center',
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: 'white',
    borderRadius: 5,
    padding: 10,
    marginRight: 10,
    backgroundColor: 'white',
    color: 'black',
  }, 
  inputNumber: {
    borderWidth: 1,
    borderColor: 'white',
    borderRadius: 5,
    padding: 10,
    marginRight: 10,
    backgroundColor: 'white',
    color: 'black',
    width: 50,
    marginBottom: 32,
    textAlignVertical: 'center',
    marginLeft: 10,
    textAlign: 'center',
  },
  addButton: {
    backgroundColor: 'cadetblue',
    padding: 10,
    borderRadius: 5,
    justifyContent: 'center',
    alignItems: 'center',
  },
  startButton: {
    backgroundColor: 'tomato',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  startButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  // picker stuff
  pickercontainer: {
    paddingBottom: 10,
  },
  dropdown: {
    height: 50,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
    elevation: 2,
  },
  placeholderStyle: {
    fontSize: 16,
    color: 'black',
  },
  selectedTextStyle: {
    fontSize: 14,
    color: 'black',
  },
  iconStyle: {
    width: 20,
    height: 20,
  },
  inputSearchStyle: {
    height: 40,
    fontSize: 16,
    color: 'black',
  },
  icon: {
    marginRight: 5,
  },
  item: {
    padding: 17,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    color: 'black',
  },
  selectedStyle: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 14,
    backgroundColor: 'white',
    shadowColor: '#000',
    marginTop: 8,
    marginRight: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
    elevation: 2,
  },
  textSelectedStyle: {
    marginRight: 5,
    fontSize: 16,
    color: 'black',
  },
});

export default StartScreen;
