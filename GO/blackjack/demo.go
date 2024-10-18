package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"slices"
)

func main() {
	var deck = [52]string{
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

	fmt.Println()

	drawn := make([]int, 0, 4)
	drawnCards := make([]int, 0, 17)
	// AFAIK 17 is the most one could need:
	// Player (A A A A 2 2 2 2 3 3 3) Dealer (3 4 4 4 4)
	drawnCards = drawCards(drawn, 4)

	fmt.Println("Player Cards: " + deck[drawnCards[0]] + " " + deck[drawnCards[1]])
	fmt.Println("Dealer Cards: " + deck[drawnCards[2]] + "  ??")

	// Player
	playerCards := []int{0, 1}
	dealerCards := []int{2, 3}
	playerHandTotal := calculateHand(playerCards, drawnCards)
	dealerHandTotal := calculateHand(dealerCards, drawnCards)

	if playerHandTotal == 21 {
		fmt.Println("Dealer Cards: " + deck[drawnCards[2]] + " " + deck[drawnCards[3]])
		if dealerHandTotal != 21 {
			fmt.Println("Player Win")
			return
		} else {
			fmt.Print("Draw")
			return
		}
	}

	if dealerHandTotal == 21 {
		fmt.Println("Dealer Cards: " + deck[drawnCards[2]] + " " + deck[drawnCards[3]])
		fmt.Print("Dealer Win")
		return
	}

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

	// Dealer
	fmt.Println("Dealer Cards: " + deck[drawnCards[2]] + " " + deck[drawnCards[3]])

	for dealerHandTotal < 17 {
		drawnCards = drawCards(drawnCards, 1)
		dealerCards = append(dealerCards, index)
		playerExtraDraw := len(playerCards) - 2

		fmt.Print("\nDealer Cards: ")
		n := len(drawnCards)
		for i := 0; i < n; i++ {
			if i == 0 || i == 1 {
				continue
			}
			if playerExtraDraw > 0 && i > 3 && playerExtraDraw+4 > i {
				continue
			}
			fmt.Print(deck[drawnCards[i]], " ")
		}
		index++
		dealerHandTotal = calculateHand(dealerCards, drawnCards)
	}
	if dealerHandTotal > 21 {
		fmt.Println(" Loose with:", dealerHandTotal, "\n")
		return
	}
	if playerHandTotal > dealerHandTotal {
		fmt.Println(" Player Win:", playerHandTotal, ">", dealerHandTotal)
	}
	if playerHandTotal == dealerHandTotal {
		fmt.Println(" Draw:", playerHandTotal, dealerHandTotal)
	}
	if playerHandTotal < dealerHandTotal {
		fmt.Println(" Dealer Win:", playerHandTotal, "<", dealerHandTotal)
	}
}

func drawCards(drawn []int, numberToDraw int) []int {
	successfulDraw := 0
	for successfulDraw < numberToDraw {
		randomNumber := rand.Intn(52)
		if slices.Contains(drawn, randomNumber) {
			// no op
		} else {
			drawn = append(drawn, randomNumber)
			successfulDraw++
		}
	}
	return drawn
}

func calculateHand(playerCards []int, drawnCards []int) int {
	sum := 0
	ace := 0
	for j := range playerCards {
		i := playerCards[j]
		card := drawnCards[i]
		if card >= 48 {
			sum += 11
			ace++
			continue
		}
		if card >= 32 {
			sum += 10
			continue
		}
		if card >= 28 {
			sum += 9
			continue
		}
		if card >= 24 {
			sum += 8
			continue
		}
		if card >= 20 {
			sum += 7
			continue
		}
		if card >= 16 {
			sum += 6
			continue
		}
		if card >= 12 {
			sum += 5
			continue
		}
		if card >= 8 {
			sum += 4
			continue
		}
		if card >= 4 {
			sum += 3
			continue
		}
		if card >= 0 {
			sum += 2
			continue
		}
	}
	for ace > 0 && sum > 21 {
		ace--
		sum -= 10
	}
	return sum
}
