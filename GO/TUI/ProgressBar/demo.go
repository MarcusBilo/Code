package main

import (
	"fmt"
	"github.com/charmbracelet/bubbles/progress"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"os"
	"strings"
	"time"
)

type tickMsg time.Time

type model struct {
	percent  float64
	progress progress.Model
}

var helpStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#626262")).Render
var pad = "\n" + strings.Repeat(" ", 2)

func main() {
	prog := progress.New(progress.WithScaledGradient("#FF7CCB", "#FDFF8C"))
	_, err := tea.NewProgram(model{progress: prog}).Run()
	if err != nil {
		fmt.Println("Oh no!", err)
		os.Exit(1)
	}
}

func (m model) View() string {
	return pad + m.progress.ViewAs(m.percent) + pad + helpStyle("Press any key to quit")
}

func (m model) Init() tea.Cmd {
	return tickCmd()
}

func tickCmd() tea.Cmd {
	tickCommand := tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
	return tickCommand
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg.(type) {

	case tea.KeyMsg:
		return m, tea.Quit

	case tickMsg:
		m.percent += 0.10
		if m.percent >= 1.0 {
			m.percent = 1.0
			return m, tea.Quit
		}
		return m, tickCmd()

	default:
		return m, nil
	}
}
