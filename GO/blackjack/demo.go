package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"slices"
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

	// Initialize

	drawnCards := make([]int, 0, 17) // As 17 is the most one round could need: (A A A A 2 2 2 2 3 3 3) & (3 4 4 4 4)
	drawnCards = drawCards(drawnCards, 4)
	playerCards := make([]int, 0, 11) // As 11 is the most the player could need: (A A A A 2 2 2 2 3 3 3)
	dealerCards := make([]int, 0, 11) // As 11 is the most the dealer could need: (A A A A 2 2 2 2 3 3 3)
	playerCards = append(playerCards, 0, 1)
	dealerCards = append(dealerCards, 2, 3)
	fmt.Println("\nPlayer Cards: " + deck[drawnCards[0]] + " " + deck[drawnCards[1]])
	fmt.Println("Dealer Cards: " + deck[drawnCards[2]] + "  ??")

	// Natural Blackjack

	playerHandTotal := calculateHand(playerCards, drawnCards)
	dealerHandTotal := calculateHand(dealerCards, drawnCards)

	if playerHandTotal == 21 || dealerHandTotal == 21 {
		fmt.Println("Dealer Cards:", deck[drawnCards[2]], deck[drawnCards[3]])
		if playerHandTotal == 21 && dealerHandTotal != 21 {
			fmt.Println("\nPlayer Win")
			return
		}
		if playerHandTotal == 21 && dealerHandTotal == 21 {
			fmt.Println("\nDraw")
			return
		}
		if playerHandTotal != 21 && dealerHandTotal == 21 {
			fmt.Println("\nDealer Win")
			return
		}
	}

	// Player Turn

	index := 4
	for playerHandTotal < 22 {
		fmt.Print("h for hit: ")
		input := bufio.NewScanner(os.Stdin)
		input.Scan()
		if input.Text() != "h" {
			break
		} else {
			drawnCards = drawCards(drawnCards, 1)
			playerCards = append(playerCards, index)

			fmt.Print("\nPlayer Cards: ")
			n := len(drawnCards)
			for i := 0; i < n; i++ {
				if i == 2 || i == 3 {
					continue
				}
				fmt.Print(deck[drawnCards[i]], " ")
			}

			index++
			playerHandTotal = calculateHand(playerCards, drawnCards)
		}
	}

	if playerHandTotal > 21 {
		fmt.Println(" Loose with:", playerHandTotal, "\n")
		return
	}

	// Dealer Turn

	fmt.Println("Dealer Cards:", deck[drawnCards[2]], deck[drawnCards[3]])

	for dealerHandTotal < 17 {
		drawnCards = drawCards(drawnCards, 1)
		dealerCards = append(dealerCards, index)
		playerExtraDraw := len(playerCards) - 2

		fmt.Print("\nDealer Cards: ")
		n := len(drawnCards)
		for k := 0; k < n; k++ {
			if k == 0 || k == 1 {
				continue
			}
			if playerExtraDraw > 0 && k > 3 && playerExtraDraw+4 > k {
				continue
			}
			fmt.Print(deck[drawnCards[k]], " ")
		}
		index++
		dealerHandTotal = calculateHand(dealerCards, drawnCards)
	}
	if dealerHandTotal > 21 {
		fmt.Println("\nLoose with:", dealerHandTotal)
		return
	}

	// Determine Winner

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

func drawCards(drawn []int, numberToDraw int) []int {
	successfulDraw := 0
	for successfulDraw < numberToDraw {
		randomNumber := rand.Intn(52)
		if !slices.Contains(drawn, randomNumber) {
			drawn = append(drawn, randomNumber)
			successfulDraw++
		}
	}
	return drawn
}

func calculateHand(inHandCards []int, drawnCards []int) int {
	sum := 0
	ace := 0
	for j := 0; j < len(inHandCards); j++ {
		card := drawnCards[inHandCards[j]]
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
