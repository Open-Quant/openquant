// TODO: implement the update_balance method into market_making logic so that when our filled orders = 0 {push balance to pf.self.balance}
use rust_decimal::prelude::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::messaging::response::Balances;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Bal {
    Symbol(String),
    Value(Decimal),
}

#[derive(Debug, Clone)]
pub struct Trade {
    side: String,
    price: f64,
}

// Struct for managing the values associated with the account
#[derive(Debug, Clone)]
pub struct Portfolio {
    id: String,
    symbol: String,
    timestamp: String,
    shares: i32,
    balance: Option<Balances>,
    trades: Vec<Trade>,
}
// Used to return simple trade metrics
impl Portfolio {
    
    // Initialize the of the Portfolio
    pub fn new() -> Portfolio {
        Portfolio {
            id: String::from("String"), // this will be the party id to identify the bot that is trading
            symbol: String::from("String"), // symbol (BEB etc)
            timestamp: String::from("m"), // timestamp of the trade
            shares: 0, // this is the number of bal of current shares
            balance: Some(Vec::new()), // this is a list of balances (Decimal) that can be used to calc return
            trades: Vec::new(), // this will be a stored list of the price of trades that occur during a session (starts off empty)
        }
    }

    // Stores the balance recevied from Erisx in Self.balance
    pub fn update_balance(&mut self, bal_update: Vec<HashMap<String, Bal>>) { //accept (type) a hashmap self.balance.push(hashmap)
        self.balance = Some(bal_update);
    }

    // TODO: calculate the change in account balance for each item
    // Method for calculating the change between the first and last value in vec(balances) as a percent(f64) 
    pub fn session_return(&mut self) {
        match self.balance.clone() { //.clone()
            Some(bal) => {
                for map in bal {
                    //self.trades.push(curreny_return); // this is the return of the currency 
                    println!("map: {:?}", map);

                    for (string, decimal) in map {
                        println!("{}: {:?}", string, decimal);
                    }
                }
            },
            None => println!("Balance is empty!"),
        }
    }
}

#[cfg(test)]
mod tests {
  use super::*;
    use std::collections::HashMap;

    fn get_test_balances() -> Vec<HashMap<String, Decimal>> {
        let mut test_balances = HashMap::new();
        test_balances.insert("availableBalance".to_string(), Decimal::from(100));
        test_balances.insert("availableBalanceCurrency".to_string(), Decimal::from(105));
        vec![test_balances]
    }
    #[test]
    fn test_make_portfolio() {
        // create pf  
        let mut pf = Portfolio::new();

        // Create hashmaps for testing
        let mut test_balances1: HashMap<String, Bal> = HashMap::new();
        test_balances1.insert("availableBalance".to_string(), Bal::Value(Decimal::from(103)));  
        test_balances1.insert("availableBalanceCurrency".to_string(), Bal::Symbol("USD".to_string()));   

        let mut test_balances2:HashMap<String, Bal> = HashMap::new();
        test_balances2.insert("availableBalance".to_string(), Bal::Value(Decimal::from(107)));   
        test_balances2.insert("availableBalanceCurrency".to_string(), Bal::Symbol("ETH".to_string()));    
        
        // Instantiate a vector to store the maps
        let mut my_vector = vec![];

        // Add maps to our vector
        my_vector.push(test_balances1);
        my_vector.push(test_balances2);

        // Call pf.update_balance
        pf.update_balance(my_vector.clone()); //.clone()

        // Call pf.session_return
        pf.session_return();

        // Print
        println!("{:#?}", &pf);
    }
    #[test]
    fn test_update_balance() {
        let mut pf = Portfolio::new();
        let test_balances = get_test_balances();
        for bal in test_balances {
            println!("{:?}", bal);
        
        }
    }
}