package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
)

func main() {
	deck := [52]string{
		" 2♥", " 2♦", " 2♣", " 2♠", // 0-3
		" 3♥", " 3♦", " 3♣", " 3♠", // 4-7
		" 4♥", " 4♦", " 4♣", " 4♠", // 8-11
		" 5♥", " 5♦", " 5♣", " 5♠", // 12-15
		" 6♥", " 6♦", " 6♣", " 6♠", // 16-19
		" 7♥", " 7♦", " 7♣", " 7♠", // 20-23
		" 8♥", " 8♦", " 8♣", " 8♠", // 24-27
		" 9♥", " 9♦", " 9♣", " 9♠", // 28-31
		"10♥", "10♦", "10♣", "10♠", // 32-35
		" J♥", " J♦", " J♣", " J♠", // 36-39
		" Q♥", " Q♦", " Q♣", " Q♠", // 40-43
		" K♥", " K♦", " K♣", " K♠", // 44-47
		" A♥", " A♦", " A♣", " A♠"} // 48-51

	// Initialize ---------------------------------------------------------------------------------------------------

	drawableCards := make([]bool, 52)
	for i := range drawableCards {
		drawableCards[i] = true
	}
	playerCards := make([]int, 0, 11) // 11 is the most the player could need: (A A A A 2 2 2 2 3 3 3)
	dealerCards := make([]int, 0, 10) // 10 is the most the dealer could need: (A A A A 2 2 2 2 3 6)
	for range []int{1, 2} {
		playerCards, drawableCards = drawOne(playerCards, drawableCards)
		dealerCards, drawableCards = drawOne(dealerCards, drawableCards)
	}
	fmt.Println("\nPlayer Cards: " + deck[playerCards[0]] + " " + deck[playerCards[1]])
	fmt.Println("Dealer Cards: " + deck[dealerCards[0]] + "  ??")

	// Natural Blackjack --------------------------------------------------------------------------------------------

	playerHandTotal := calculateHand(playerCards)
	dealerHandTotal := calculateHand(dealerCards)

	if playerHandTotal == 21 || dealerHandTotal == 21 {
		fmt.Println("Dealer Cards:", deck[dealerCards[0]], deck[dealerCards[1]])
		if playerHandTotal == 21 && dealerHandTotal != 21 {
			fmt.Println("\nNatural Blackjack - Player Win")
			return
		}
		if playerHandTotal == 21 && dealerHandTotal == 21 {
			fmt.Println("\nNatural Blackjack - Draw")
			return
		}
		if playerHandTotal != 21 && dealerHandTotal == 21 {
			fmt.Println("\nNatural Blackjack - Dealer Win")
			return
		}
	}

	// Player Turn --------------------------------------------------------------------------------------------------

	for playerHandTotal < 22 {
		fmt.Print("h for hit, anything else to stand: ")
		input := bufio.NewScanner(os.Stdin)
		input.Scan()
		if input.Text() != "h" {
			break
		} else {

			playerCards, drawableCards = drawOne(playerCards, drawableCards)

			fmt.Print("\nPlayer Cards: ")
			for i := range playerCards {
				fmt.Print(deck[playerCards[i]], " ")
			}

			playerHandTotal = calculateHand(playerCards)
		}
	}

	if playerHandTotal > 21 {
		fmt.Println(" Player Loose with:", playerHandTotal)
		return
	}

	// Dealer Turn --------------------------------------------------------------------------------------------------

	fmt.Println("Dealer Cards:", deck[dealerCards[0]], deck[dealerCards[1]])

	for dealerHandTotal < 17 {

		dealerCards, drawableCards = drawOne(dealerCards, drawableCards)

		fmt.Print("\nDealer Cards: ")

		for i := range dealerCards {
			fmt.Print(deck[dealerCards[i]], " ")
		}

		dealerHandTotal = calculateHand(dealerCards)
	}
	if dealerHandTotal > 21 {
		fmt.Println("\nDealer Loose with:", dealerHandTotal)
		return
	}

	// Determine Winner ---------------------------------------------------------------------------------------------

	playerWin := fmt.Sprintf("\nPlayer Win: %d > %d", playerHandTotal, dealerHandTotal)
	neitherWin := fmt.Sprintf("\nNeither Win: %d = %d", playerHandTotal, dealerHandTotal)
	dealerWin := fmt.Sprintf("\nDealer Win: %d < %d", playerHandTotal, dealerHandTotal)

	if playerHandTotal > dealerHandTotal {
		fmt.Println(playerWin)
	} else if playerHandTotal == dealerHandTotal {
		fmt.Println(neitherWin)
	} else if playerHandTotal < dealerHandTotal {
		fmt.Println(dealerWin)
	}

}

func drawOne(alreadyDrawnCards []int, remainingPoolOfCards []bool) ([]int, []bool) {

	randomlyDrawnCard := rand.Intn(len(remainingPoolOfCards))

	for remainingPoolOfCards[randomlyDrawnCard] == false {
		randomlyDrawnCard = rand.Intn(len(remainingPoolOfCards))
	}

	remainingPoolOfCards[randomlyDrawnCard] = false

	alreadyDrawnCards = append(alreadyDrawnCards, randomlyDrawnCard)

	return alreadyDrawnCards, remainingPoolOfCards
}

func calculateHand(cardsInHand []int) int {
	sum := 0
	ace := 0
	for i := range cardsInHand {
		card := cardsInHand[i]
		switch {
		case card >= 48:
			sum += 11
			ace++
		case card >= 32:
			sum += 10
		case card >= 28:
			sum += 9
		case card >= 24:
			sum += 8
		case card >= 20:
			sum += 7
		case card >= 16:
			sum += 6
		case card >= 12:
			sum += 5
		case card >= 8:
			sum += 4
		case card >= 4:
			sum += 3
		case card >= 0:
			sum += 2
		}
	}
	for ace > 0 && sum > 21 {
		sum -= 10
		ace--
	}
	return sum
}
